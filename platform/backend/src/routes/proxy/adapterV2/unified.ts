/**
 * Unified LLM Proxy Adapter
 *
 * Auto-routes to any LLM provider based on the model name.
 * Responses are always normalized to OpenAI format.
 *
 * Routing strategy:
 *  1. Explicit provider prefix in model (e.g. "anthropic/claude-3-5-sonnet")
 *  2. Pattern matching on model name keywords
 *  3. Default: OpenAI
 */

import { get } from "lodash-es";
import type { SupportedProvider } from "@shared";
import type {
  CreateClientOptions,
  GenAiOperationName,
  LLMProvider,
  LLMRequestAdapter,
  LLMResponseAdapter,
  LLMStreamAdapter,
  OpenAi,
} from "@/types";
import {
  OpenAIRequestAdapter,
  OpenAIResponseAdapter,
  OpenAIStreamAdapter,
} from "./openai";
import { anthropicAdapterFactory } from "./anthropic";
import { bedrockAdapterFactory } from "./bedrock";
import { cerebrasAdapterFactory } from "./cerebras";
import { cohereAdapterFactory } from "./cohere";
import { deepseekAdapterFactory } from "./deepseek";
import { geminiAdapterFactory } from "./gemini";
import { groqAdapterFactory } from "./groq";
import { minimaxAdapterFactory } from "./minimax";
import { mistralAdapterFactory } from "./mistral";
import { ollamaAdapterFactory } from "./ollama";
import { openaiAdapterFactory } from "./openai";
import { openrouterAdapterFactory } from "./openrouter";
import { perplexityAdapterFactory } from "./perplexity";
import { vllmAdapterFactory } from "./vllm";
import { xaiAdapterFactory } from "./xai";
import { zhipuaiAdapterFactory } from "./zhipuai";

// =============================================================================
// TYPES
// =============================================================================

type UnifiedRequest = OpenAi.Types.ChatCompletionsRequest;
type UnifiedResponse = OpenAi.Types.ChatCompletionsResponse;
type UnifiedMessages = OpenAi.Types.ChatCompletionsRequest["messages"];
type UnifiedHeaders = OpenAi.Types.ChatCompletionsHeaders;
type UnifiedStreamChunk = OpenAi.Types.ChatCompletionChunk;

export type { UnifiedRequest, UnifiedResponse, UnifiedMessages, UnifiedHeaders, UnifiedStreamChunk };

// =============================================================================
// PROVIDER REGISTRY
// =============================================================================

interface ProviderEntry {
  factory: LLMProvider<unknown, unknown, unknown, unknown, unknown>;
  /** Whether this provider accepts OpenAI-format chat completions requests */
  openAICompatible: boolean;
}

const PROVIDERS: Record<SupportedProvider, ProviderEntry> = {
  openai:      { factory: openaiAdapterFactory,      openAICompatible: true  },
  anthropic:   { factory: anthropicAdapterFactory,    openAICompatible: false },
  gemini:      { factory: geminiAdapterFactory,       openAICompatible: false },
  cohere:      { factory: cohereAdapterFactory,       openAICompatible: false },
  mistral:     { factory: mistralAdapterFactory,      openAICompatible: true  },
  deepseek:    { factory: deepseekAdapterFactory,     openAICompatible: true  },
  xai:         { factory: xaiAdapterFactory,          openAICompatible: true  },
  groq:        { factory: groqAdapterFactory,         openAICompatible: true  },
  cerebras:    { factory: cerebrasAdapterFactory,    openAICompatible: true  },
  perplexity:  { factory: perplexityAdapterFactory,  openAICompatible: true  },
  openrouter:  { factory: openrouterAdapterFactory,  openAICompatible: true  },
  vllm:        { factory: vllmAdapterFactory,         openAICompatible: true  },
  ollama:      { factory: ollamaAdapterFactory,       openAICompatible: true  },
  zhipuai:     { factory: zhipuaiAdapterFactory,      openAICompatible: false },
  bedrock:     { factory: bedrockAdapterFactory,      openAICompatible: false },
  minimax:     { factory: minimaxAdapterFactory,      openAICompatible: true  },
};

// =============================================================================
// MODEL → PROVIDER ROUTING
// =============================================================================

function resolveProvider(model: string): SupportedProvider {
  const m = model.toLowerCase();

  // Explicit provider prefixes
  if (m.startsWith("anthropic/"))   return "anthropic";
  if (m.startsWith("google/"))      return "gemini";
  if (m.startsWith("cohere/"))      return "cohere";
  if (m.startsWith("mistral/"))     return "mistral";
  if (m.startsWith("deepseek/"))    return "deepseek";
  if (m.startsWith("x-ai/"))        return "xai";
  if (m.startsWith("perplexity/"))  return "perplexity";
  if (m.startsWith("groq/"))        return "groq";
  if (m.startsWith("cerebras/"))    return "cerebras";
  if (m.startsWith("openrouter/"))  return "openrouter";
  if (m.startsWith("vllm/"))        return "vllm";
  if (m.startsWith("ollama/"))      return "ollama";
  if (m.startsWith("zhipuai/"))     return "zhipuai";
  if (m.startsWith("bedrock/"))     return "bedrock";
  if (m.startsWith("minimax/"))     return "minimax";

  // Keyword-based routing
  if (m.includes("claude"))          return "anthropic";
  if (m.includes("gemini"))          return "gemini";
  if (m.includes("command-r") || m.includes("command_") || m.includes("coval"))
                                           return "cohere";
  if (m.includes("mistral") || m.includes("mixtral") || m.includes("codestral"))
                                           return "mistral";
  if (m.includes("deepseek"))        return "deepseek";
  if (m.includes("grok"))            return "xai";
  if (m.includes("sonar"))           return "perplexity";
  if (m.includes("minimax") || m.includes("abab"))
                                           return "minimax";
  if (m.includes("cerebras"))        return "cerebras";
  if (m.includes("zhipu") || m.includes("glm-"))
                                           return "zhipuai";
  if (m.includes("amazon") || m.includes("meta.llama") || m.includes("ai21"))
                                           return "bedrock";

  // Default
  return "openai";
}

export { resolveProvider as resolveProviderForModel };

// =============================================================================
// UNIFIED REQUEST ADAPTER
// =============================================================================

class UnifiedRequestAdapter
  implements LLMRequestAdapter<UnifiedRequest, UnifiedMessages> {
  readonly provider: SupportedProvider;
  private delegate: OpenAIRequestAdapter;

  constructor(request: UnifiedRequest) {
    this.provider = resolveProvider(request.model ?? "");
    // All unified requests arrive in OpenAI format; the upstream SDK
    // (or the proxy pre-processing) handles any needed conversion.
    this.delegate = new OpenAIRequestAdapter(request);
  }

  getModel()                   { return this.delegate.getModel(); }
  isStreaming()                { return this.delegate.isStreaming(); }
  getMessages()                { return this.delegate.getMessages(); }
  getProviderMessages()        { return this.delegate.getProviderMessages() as UnifiedMessages; }
  hasTools()                   { return this.delegate.hasTools(); }
  getTools()                   { return this.delegate.getTools(); }
  getNativeTools()             { return this.delegate.getNativeTools?.() ?? []; }
  setModel(model: string)      { this.delegate.setModel(model); }
  clone()                      { return this; }
}

// =============================================================================
// UNIFIED STREAM ADAPTER
// =============================================================================

/**
 * The OpenAIStreamAdapter already normalizes all chunks to OpenAI SSE format.
 * This is exactly the format the unified endpoint should return.
 */
class UnifiedStreamAdapter extends OpenAIStreamAdapter {}

// =============================================================================
// UNIFIED CLIENT
// =============================================================================

class UnifiedClient {
  constructor(
    private apiKey: string | undefined,
    private options: CreateClientOptions,
  ) {}

  async execute(request: UnifiedRequest): Promise<UnifiedResponse> {
    const provider = resolveProvider(request.model ?? "");
    const entry = PROVIDERS[provider];
    const client = entry.factory.createClient(this.apiKey, this.options);
    const result = await entry.factory.execute(client, request);

    // OpenAI-compatible providers return OpenAI-format responses directly
    if (entry.openAICompatible) {
      return result as UnifiedResponse;
    }

    // Non-OpenAI providers: extract normalized data via response adapter
    const respAdapter = entry.factory.createResponseAdapter(result);
    return (respAdapter as unknown as { getOriginalResponse(): UnifiedResponse })
      .getOriginalResponse();
  }

  async executeStream(request: UnifiedRequest) {
    const provider = resolveProvider(request.model ?? "");
    const entry = PROVIDERS[provider];
    const client = entry.factory.createClient(this.apiKey, this.options);
    return entry.factory.executeStream(client, request) as Promise<AsyncIterable<UnifiedStreamChunk>>;
  }
}

// =============================================================================
// UNIFIED ADAPTER FACTORY
// =============================================================================

export const unifiedAdapterFactory: LLMProvider<
  UnifiedRequest,
  UnifiedResponse,
  UnifiedMessages,
  UnifiedStreamChunk,
  UnifiedHeaders
> = {
  provider: "openai",
  interactionType: "openai:chatCompletions",

  createRequestAdapter(request: UnifiedRequest) {
    return new UnifiedRequestAdapter(request) as unknown as LLMRequestAdapter<UnifiedRequest, UnifiedMessages>;
  },

  createResponseAdapter(response: UnifiedResponse) {
    return new OpenAIResponseAdapter(response) as unknown as LLMResponseAdapter<UnifiedResponse>;
  },

  createStreamAdapter() {
    return new UnifiedStreamAdapter() as unknown as LLMStreamAdapter<UnifiedStreamChunk, UnifiedResponse>;
  },

  extractApiKey(headers: UnifiedHeaders) {
    const h = headers as Record<string, string | undefined>;
    const auth = h["authorization"] ?? h["Authorization"];
    if (typeof auth === "string" && auth.startsWith("Bearer ")) {
      return auth.slice(7);
    }
    return undefined;
  },

  getBaseUrl() {
    return undefined; // Use SDK default
  },

  spanName: "chat" as GenAiOperationName,

  createClient(apiKey: string | undefined, options: CreateClientOptions) {
    return new UnifiedClient(apiKey, options);
  },

  async execute(client: unknown, request: UnifiedRequest) {
    return (client as UnifiedClient).execute(request);
  },

  async executeStream(client: unknown, request: UnifiedRequest) {
    return (client as UnifiedClient).executeStream(request);
  },

  extractErrorMessage(error: unknown) {
    const msg = get(error, "error.message");
    if (typeof msg === "string") return msg;
    if (error instanceof Error) return error.message;
    return "Internal server error";
  },
};
