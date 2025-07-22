import {
  Application,
  Router,
  Context,
  Status,
  isHttpError,
} from "https://deno.land/x/oak@v12.6.1/mod.ts";
import { crypto } from "https://deno.land/std@0.192.0/crypto/mod.ts"; // For SHA256
import { toHashString } from "https://deno.land/std@0.192.0/crypto/to_hash_string.ts"; // Helper for hex digest

// --- Configuration ---
const AVAILABLE_MODELS = [
  "claude-3-7-sonnet",
  "claude-3-5-haiku",
  "deepseek-v3",
  "deepseek-r1",
  "o3-mini",
  "gpt-4.1",
  "gpt-4.1-mini",
];

const THETAWAVE_API_BASE = "https://thetawave.ai/api";
const THETAWAVE_STREAM_URL = "https://thetawave.ai/api/http/stream/v1";
const USER_AGENT =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0";

// --- Simple LRU Cache Implementation ---
class SimpleLRUCache<K, V> {
  private cache: Map<K, V>;
  private maxSize: number;

  constructor(maxSize: number) {
    this.cache = new Map<K, V>();
    this.maxSize = maxSize;
  }

  get(key: K): V | undefined {
    const value = this.cache.get(key);
    if (value) {
      // Move to end (most recently used)
      this.cache.delete(key);
      this.cache.set(key, value);
    }
    return value;
  }

  set(key: K, value: V): void {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    } else if (this.cache.size >= this.maxSize) {
      // Evict least recently used (first item in map iteration)
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) {
        this.cache.delete(firstKey);
      }
    }
    this.cache.set(key, value);
  }
}

const CHAT_ID_CACHE = new SimpleLRUCache<string, { chatId: string; userId: string }>(100);

// --- TypeScript Interfaces (Equivalent to Pydantic Models) ---
interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
  reasoning_content?: string | null;
  // ThetaWave specific fields added for internal use
  parts?: Array<{ type: string; text: string }>;
  annotations?: { sources: any[] };
}

interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  stream?: boolean;
  temperature?: number | null;
  max_tokens?: number | null;
  top_p?: number | null;
}

interface ModelInfo {
  id: string;
  object: "model";
  created: number;
  owned_by: string;
}

interface ModelList {
  object: "list";
  data: ModelInfo[];
}

interface ChatCompletionChoice {
  message: ChatMessage;
  index: number;
  finish_reason: string;
}

interface ChatCompletionResponse {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface StreamChoice {
  delta: Record<string, any>;
  index: number;
  finish_reason?: string | null;
}

interface StreamResponse {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: StreamChoice[];
}

// --- Helper Functions ---

async function sha256Hex(input: string): Promise<string> {
    const data = new TextEncoder().encode(input);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    return toHashString(hashBuffer);
}

function getCurrentTimestamp(): number {
  return Math.floor(Date.now() / 1000);
}

function generateChatId(): string {
  return `chatcmpl-${crypto.randomUUID().replace(/-/g, "")}`;
}

function getModelsListResponse(): ModelList {
  const modelInfos: ModelInfo[] = AVAILABLE_MODELS.map((modelId) => ({
    id: modelId,
    object: "model",
    created: getCurrentTimestamp(),
    owned_by: "ThetaWave",
  }));
  return { object: "list", data: modelInfos };
}

async function createThetawaveChat(authSession: string): Promise<{ chatId: string; userId: string }> {
  const url = `${THETAWAVE_API_BASE}/rag.chat.createChat?batch=1`;
  const payload = { "0": { json: { title: "Assistant" } } };
  const headers = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Content-Type": "application/json",
    "origin": "https://thetawave.ai",
    "referer": "https://thetawave.ai/app/chat",
    "Cookie": `auth_session=${authSession}`,
  };

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: headers,
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`ThetaWave createChat API error (${response.status}): ${errorText}`);
      throw new Error(`Failed to create ThetaWave chat: ${response.status} ${errorText}`);
    }

    const data = await response.json();
    if (!data || !data[0]?.result?.data?.json) {
        throw new Error("Invalid response structure from createChat API");
    }
    const chatId = data[0].result.data.json.id;
    const userId = data[0].result.data.json.userId;
    if (!chatId || !userId) {
        throw new Error("Missing chatId or userId in createChat response");
    }
    return { chatId, userId };
  } catch (error) {
    console.error("Error during createThetawaveChat:", error);
    throw error; // Re-throw after logging
  }
}

async function getOrCreateChatId(authSession: string, messages: ChatMessage[]): Promise<{ chatId: string; userId: string }> {
    if (!messages || messages.length === 0) {
        console.log("No messages provided, creating new chat.");
        return await createThetawaveChat(authSession);
    }

    const firstUserMessage = messages.find(msg => msg.role === "user");

    if (!firstUserMessage || !firstUserMessage.content) {
        console.log("No user message found, creating new chat.");
        return await createThetawaveChat(authSession);
    }

    // Create a stable key based on session and first user message content
    const conversationKey = await sha256Hex(`${authSession}:${firstUserMessage.content}`);

    const cachedData = CHAT_ID_CACHE.get(conversationKey);
    if (cachedData) {
        console.log("Found existing chat ID in cache.");
        return cachedData;
    }

    console.log("No cached chat ID found, creating new chat.");
    const newData = await createThetawaveChat(authSession);
    CHAT_ID_CACHE.set(conversationKey, newData);
    return newData;
}


async function* streamThetawaveResponseParser(
    responseBody: ReadableStream<Uint8Array>,
    model: string
): AsyncGenerator<string> {
    const streamId = generateChatId();
    const createdTime = getCurrentTimestamp();
    const decoder = new TextDecoder();
    const reader = responseBody.getReader();

    // Send initial role delta
    const initialDelta: StreamResponse = {
        id: streamId,
        created: createdTime,
        model: model,
        object: "chat.completion.chunk",
        choices: [{ delta: { role: "assistant" }, index: 0 }],
    };
    yield `data: ${JSON.stringify(initialDelta)}\n\n`;

    let buffer = "";
    let reasoningBuffer = ""; // To accumulate reasoning content if needed later
    let contentBuffer = "";   // To accumulate actual content if needed later

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                console.log("Stream finished from reader.");
                break;
            }

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || ""; // Keep the last partial line

            for (const line of lines) {
                if (!line.trim()) continue;

                let delta: Record<string, any> = {};
                let finishReason: string | null = null;
                let isEndMarker = false;

                try {
                    if (line.startsWith('g:')) { // Thinking/reasoning content
                        let content = line.substring(2);
                        try {
                            content = JSON.parse(content);
                        } catch {
                            // Handle plain text like g:"text"
                            if (content.startsWith('"') && content.endsWith('"')) {
                                content = content.slice(1, -1);
                            }
                        }
                        reasoningBuffer += content; // Accumulate if needed
                        delta = { reasoning_content: content };
                    } else if (line.startsWith('0:')) { // Actual content
                        let content = line.substring(2);
                        try {
                            content = JSON.parse(content);
                        } catch {
                             // Handle plain text like 0:"text"
                            if (content.startsWith('"') && content.endsWith('"')) {
                                content = content.slice(1, -1);
                            }
                        }
                        contentBuffer += content; // Accumulate if needed
                        delta = { content: content };
                    } else if (line.startsWith('e:') || line.startsWith('d:')) { // End markers
                        finishReason = 'stop';
                        isEndMarker = true;
                    } else {
                        console.warn("Unknown stream line format:", line);
                        continue; // Skip unknown lines
                    }

                    const streamChunk: StreamResponse = {
                        id: streamId,
                        created: createdTime,
                        model: model,
                        object: "chat.completion.chunk",
                        choices: [{ delta: delta, index: 0, finish_reason: finishReason }],
                    };
                    yield `data: ${JSON.stringify(streamChunk)}\n\n`;

                    if (isEndMarker) {
                        console.log("End marker received, sending [DONE].");
                        yield "data: [DONE]\n\n";
                        return; // Exit generator after end marker
                    }
                } catch (parseError) {
                    console.error("Error parsing stream line:", line, parseError);
                }
            }
        }
    } catch (error) {
        console.error("Error reading stream:", error);
        // Yield an error message in the stream if possible?
        const errorChunk: StreamResponse = {
             id: streamId,
             created: createdTime,
             model: model,
             object: "chat.completion.chunk",
             choices: [{ delta: { content: `\n\n[STREAM ERROR: ${error.message}]`}, index: 0, finish_reason: "error" }],
        };
         yield `data: ${JSON.stringify(errorChunk)}\n\n`;
    } finally {
        // Ensure [DONE] is sent if the loop finishes without an end marker
        console.log("Stream loop finished or errored, ensuring [DONE] is sent.");
        yield "data: [DONE]\n\n";
        // Release the lock, though exiting the generator should handle this.
        // reader.releaseLock(); // Not strictly necessary as generator exit handles it
    }
}


async function buildNonStreamThetawaveResponse(
    responseBody: ReadableStream<Uint8Array>,
    model: string
): Promise<ChatCompletionResponse> {
    const decoder = new TextDecoder();
    const reader = responseBody.getReader();
    let reasoningContent = "";
    let content = "";
    let buffer = "";

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || ""; // Keep potential partial line

            for (const line of lines) {
                 if (!line.trim()) continue;

                 try {
                    if (line.startsWith('g:')) {
                        let text = line.substring(2);
                        try { text = JSON.parse(text); } catch {
                            if (text.startsWith('"') && text.endsWith('"')) text = text.slice(1, -1);
                        }
                        reasoningContent += text;
                    } else if (line.startsWith('0:')) {
                        let text = line.substring(2);
                        try { text = JSON.parse(text); } catch {
                             if (text.startsWith('"') && text.endsWith('"')) text = text.slice(1, -1);
                        }
                        content += text;
                    } else if (line.startsWith('e:') || line.startsWith('d:')) {
                        // End marker found, stop processing
                        buffer = ""; // Clear buffer as we are done
                        break; // Exit inner loop
                    }
                 } catch (parseError) {
                     console.error("Error parsing non-stream line:", line, parseError);
                     // Check if we broke from the inner loop due to end marker
                     if (line.startsWith('e:') || line.startsWith('d:')) {
                         break; // Exit inner loop immediately after finding end marker
                     }
                }
                // If the inner loop finished because of an end marker, break the outer loop too.
                if (buffer === "" && (line?.startsWith('e:') || line?.startsWith('d:'))) { // Check if the last processed line was an end marker
                     break; // Exit outer while loop
                }
            }
        } // End of while loop
    } // End of the main try block started at line 348
    catch (error) { // Catch for the try block starting at line 348
        console.error("Error reading non-stream response:", error);
        // Depending on requirements, might throw or return partial data
        throw new Error(`Failed to fully read ThetaWave response: ${error.message}`);
    } finally {
         // reader.releaseLock(); // Not needed as loop exit/throw handles it
    }


    return {
        id: generateChatId(),
        object: "chat.completion",
        created: getCurrentTimestamp(),
        model: model,
        choices: [
            {
                message: {
                    role: "assistant",
                    content: content,
                    reasoning_content: reasoningContent || null,
                },
                index: 0,
                finish_reason: "stop", // Assume stop if stream completed normally
            },
        ],
        usage: { // Usage data is not available from ThetaWave stream, return zeros
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    };
}

// --- Oak Application Setup ---
const app = new Application();
const router = new Router();

// Middleware for Authentication and Error Handling
app.use(async (ctx, next) => {
  try {
    await next();
  } catch (err) {
    if (isHttpError(err)) {
      ctx.response.status = err.status;
      ctx.response.body = { error: { message: err.message, type: "request_error", code: err.status } };
       console.error(`HTTP Error (${err.status}): ${err.message}`);
    } else {
      ctx.response.status = Status.InternalServerError;
      ctx.response.body = { error: { message: "Internal Server Error", type: "internal_error", code: 500 } };
      console.error("Internal Server Error:", err);
    }
  }
});

// Authentication Middleware
const authMiddleware = async (ctx: Context, next: () => Promise<unknown>) => {
  const authHeader = ctx.request.headers.get("Authorization");
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    ctx.throw(Status.Unauthorized, "ThetaWave auth_session required in Authorization header (Bearer token).");
  }
  const token = authHeader.substring(7); // Remove "Bearer "
  if (!token) {
     ctx.throw(Status.Unauthorized, "ThetaWave auth_session token is empty.");
  }
  ctx.state.authSession = token; // Store token for route handlers
  await next();
};


// --- Routes ---
router.get("/v1/models", authMiddleware, (ctx) => {
  ctx.response.body = getModelsListResponse();
});

router.get("/models", (ctx) => {
  // No auth required for compatibility
  ctx.response.body = getModelsListResponse();
});

router.post("/v1/chat/completions", authMiddleware, async (ctx) => {
  if (!ctx.request.hasBody) {
    ctx.throw(Status.BadRequest, "Request body is required.");
  }

  const body = ctx.request.body({ type: "json" });
  const requestData: ChatCompletionRequest = await body.value;
  const authSession = ctx.state.authSession as string;

  // Basic validation
  if (!requestData.model || !requestData.messages || requestData.messages.length === 0) {
      ctx.throw(Status.BadRequest, "Missing required fields: model and messages.");
  }

  if (!AVAILABLE_MODELS.includes(requestData.model)) {
    ctx.throw(Status.NotFound, `Model '${requestData.model}' not found or not available.`);
  }

  try {
    const { chatId, userId } = await getOrCreateChatId(authSession, requestData.messages);

    const userMessage = requestData.messages[requestData.messages.length - 1]?.content ?? "";
    let systemPrompt: string | null = null;
    const thetawaveMessages: ChatMessage[] = [];

    for (const msg of requestData.messages) {
        if (msg.role === "system") {
            systemPrompt = msg.content;
            continue;
        }
        if (msg.role === "user" || msg.role === "assistant") {
             const messagePayload: ChatMessage = {
                role: msg.role,
                content: msg.content,
                parts: [{ type: "text", text: msg.content }] // ThetaWave format
            };
            if (msg.role === "user") {
                messagePayload.annotations = { sources: [] }; // ThetaWave format
            }
            thetawaveMessages.push(messagePayload);
        }
    }


    const payload: any = {
      id: "0", // ThetaWave specific
      messages: thetawaveMessages,
      userId: userId,
      chatId: chatId,
      useLocalSearch: false,
      useWebSearch: false,
      selectedLocalResources: [],
      model: requestData.model,
      userMessage: userMessage, // Last user message content
    };

    if (systemPrompt) {
        payload.systemPrompt = systemPrompt;
    }

    const headers = {
      "User-Agent": USER_AGENT,
      "Accept-Encoding": "gzip, deflate, br, zstd",
      "Content-Type": "application/json",
      "origin": "https://thetawave.ai",
      "referer": "https://thetawave.ai/app/chat",
      "Cookie": `auth_session=${authSession}`,
      "Accept": "text/event-stream", // Important for streaming
    };

    const response = await fetch(THETAWAVE_STREAM_URL, {
      method: "POST",
      headers: headers,
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
        const errorText = await response.text();
        console.error(`ThetaWave API stream error (${response.status}): ${errorText}`);
        // For stream requests, we should try to send an error event if possible
        if (requestData.stream) {
             const errorPayload = { error: { message: errorText, type: "thetawave_api_error", code: response.status } };
             ctx.response.body = new ReadableStream({
                 start(controller) {
                     const message = `data: ${JSON.stringify(errorPayload)}\n\ndata: [DONE]\n\n`;
                     controller.enqueue(new TextEncoder().encode(message));
                     controller.close();
                 }
             });
             ctx.response.type = "text/event-stream";
             ctx.response.status = response.status; // Reflect the upstream error status
        } else {
            ctx.throw(response.status, `ThetaWave API Error: ${errorText}`);
        }
        return; // Stop processing if response not ok
    }

    if (requestData.stream) {
        if (!response.body) {
             ctx.throw(Status.InternalServerError, "ThetaWave stream response body is null.");
        }
        ctx.response.body = new ReadableStream({
            async start(controller) {
                const encoder = new TextEncoder();
                try {
                    for await (const chunk of streamThetawaveResponseParser(response.body!, requestData.model)) {
                        controller.enqueue(encoder.encode(chunk));
                    }
                } catch (streamError) {
                     console.error("Error processing stream:", streamError);
                     // Try to enqueue an error message if the stream hasn't closed
                     try {
                        const errorChunk = `data: ${JSON.stringify({error: {message: `Stream processing error: ${streamError.message}`, type: "internal_error", code: 500}})}\n\ndata: [DONE]\n\n`;
                        controller.enqueue(encoder.encode(errorChunk));
                     } catch (enqueueError) {
                         console.error("Failed to enqueue stream error message:", enqueueError);
                     }
                } finally {
                    try {
                        controller.close();
                    } catch (closeError) {
                         console.error("Error closing stream controller:", closeError);
                    }
                }
            }
        });
        ctx.response.type = "text/event-stream";
        ctx.response.headers.set("Cache-Control", "no-cache");
        ctx.response.headers.set("Connection", "keep-alive");
        ctx.response.headers.set("X-Accel-Buffering", "no"); // For nginx etc.
    } else {
        if (!response.body) {
             ctx.throw(Status.InternalServerError, "ThetaWave non-stream response body is null.");
             return; // Explicitly return to avoid proceeding
        }
        // Need to consume the stream fully for non-stream response
        ctx.response.body = await buildNonStreamThetawaveResponse(response.body, requestData.model);
        ctx.response.type = "application/json";
    }

  } catch (error) {
     // Catch errors from getOrCreateChatId or other upstream issues before fetch
     console.error("Error in chat completions handler:", error);
     if (requestData.stream) {
         // Send error stream for stream requests
         const errorPayload = { error: { message: error.message || "Unknown error", type: "internal_error", code: 500 } };
         ctx.response.body = new ReadableStream({
             start(controller) {
                 const message = `data: ${JSON.stringify(errorPayload)}\n\ndata: [DONE]\n\n`;
                 controller.enqueue(new TextEncoder().encode(message));
                 controller.close();
             }
         });
         ctx.response.type = "text/event-stream";
         ctx.response.status = Status.InternalServerError;
     } else {
         // Throw HTTP error for non-stream requests
         if (isHttpError(error)) {
             ctx.throw(error.status, error.message);
         } else {
             ctx.throw(Status.InternalServerError, error.message || "Internal server error during chat completion.");
         }
     }
  }
});

app.use(router.routes());
app.use(router.allowedMethods());

// --- Server Start ---
const PORT = 8000;

app.addEventListener("listen", ({ hostname, port, secure }) => {
  console.log("\n--- ThetaWave OpenAI API Adapter (Deno/Oak) ---");
  console.log("Authentication: Use 'Authorization: Bearer <your_thetawave_auth_session>'");
  console.log("Endpoints:");
  console.log(`  GET  http://${hostname ?? "localhost"}:${port}/v1/models (Auth Required)`);
  console.log(`  GET  http://${hostname ?? "localhost"}:${port}/models (No Auth)`);
  console.log(`  POST http://${hostname ?? "localhost"}:${port}/v1/chat/completions (Auth Required)`);
  console.log(`\nAvailable Models: ${AVAILABLE_MODELS.join(", ")}`);
  console.log("------------------------------------");
  console.log(
    `🚀 Server listening on: ${secure ? "https://" : "http://"}${hostname ??
      "localhost"}:${port}`
  );
});

await app.listen({ port: PORT });
