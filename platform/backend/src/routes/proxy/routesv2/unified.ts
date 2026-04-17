/**
 * Unified LLM Proxy Routes
 *
 * Provides a single `/v1/unified/...` endpoint that accepts OpenAI-format
 * requests and auto-routes them to the appropriate upstream LLM provider
 * based on the model name.
 *
 * Supported providers: OpenAI, Anthropic, Gemini, Cohere, Mistral, DeepSeek,
 * xAI, Groq, Cerebras, Perplexity, OpenRouter, vLLM, Ollama, ZhipuAI, Bedrock, MiniMax
 *
 * All responses are returned in OpenAI-compatible format.
 */

import { RouteId } from "@shared";
import type { FastifyPluginAsyncZod } from "fastify-type-provider-zod";
import { z } from "zod";
import logger from "@/logging";
import { constructResponseSchema, OpenAi, UuidIdSchema } from "@/types";
import { unifiedAdapterFactory } from "../adapterV2/unified";
import { PROXY_API_PREFIX, PROXY_BODY_LIMIT } from "../common";
import { handleLLMProxy } from "../llm-proxy-handler";

const unifiedProxyRoutes: FastifyPluginAsyncZod = async (fastify) => {
  const API_PREFIX = `${PROXY_API_PREFIX}/unified`;
  const CHAT_COMPLETIONS_SUFFIX = "/chat/completions";

  logger.info("[UnifiedProxy] Registering unified proxy routes");

  /**
   * POST /v1/unified/chat/completions
   * Unified chat completions endpoint using the default agent.
   * Model name in request body determines which upstream provider to use.
   */
  fastify.post(
    `${API_PREFIX}${CHAT_COMPLETIONS_SUFFIX}`,
    {
      bodyLimit: PROXY_BODY_LIMIT,
      schema: {
        operationId: RouteId.UnifiedChatCompletionsWithDefaultAgent,
        description:
          "Create a chat completion with any provider via unified endpoint. " +
          "The model name in the request body determines the upstream provider. " +
          "Response is always in OpenAI format.",
        tags: ["LLM Proxy"],
        body: OpenAi.API.ChatCompletionRequestSchema,
        headers: OpenAi.API.ChatCompletionsHeadersSchema,
        response: constructResponseSchema(
          OpenAi.API.ChatCompletionResponseSchema,
        ),
      },
    },
    async (request, reply) => {
      const model = (request.body as { model?: string }).model ?? "";
      logger.debug(
        { url: request.url, model },
        "[UnifiedProxy] Handling unified request (default agent)",
      );
      return handleLLMProxy(request.body, request, reply, unifiedAdapterFactory);
    },
  );

  /**
   * POST /v1/unified/:agentId/chat/completions
   * Unified chat completions endpoint for a specific agent.
   */
  fastify.post(
    `${API_PREFIX}/:agentId${CHAT_COMPLETIONS_SUFFIX}`,
    {
      bodyLimit: PROXY_BODY_LIMIT,
      schema: {
        operationId: RouteId.UnifiedChatCompletionsWithAgent,
        description:
          "Create a chat completion with any provider via unified endpoint for a specific agent. " +
          "The model name in the request body determines the upstream provider. " +
          "Response is always in OpenAI format.",
        tags: ["LLM Proxy"],
        params: z.object({
          agentId: UuidIdSchema,
        }),
        body: OpenAi.API.ChatCompletionRequestSchema,
        headers: OpenAi.API.ChatCompletionsHeadersSchema,
        response: constructResponseSchema(
          OpenAi.API.ChatCompletionResponseSchema,
        ),
      },
    },
    async (request, reply) => {
      const model = (request.body as { model?: string }).model ?? "";
      const agentId = request.params.agentId;
      logger.debug(
        { url: request.url, model, agentId },
        "[UnifiedProxy] Handling unified request (with agent)",
      );
      return handleLLMProxy(request.body, request, reply, unifiedAdapterFactory);
    },
  );

  /**
   * GET /v1/unified/models
   * Returns a list of all available models from all providers in OpenAI format.
   * This aggregates models from all configured providers.
   */
  fastify.get(
    `${API_PREFIX}/models`,
    {
      bodyLimit: PROXY_BODY_LIMIT,
      schema: {
        operationId: RouteId.UnifiedListModels,
        description:
          "List all available models from all providers in OpenAI format. " +
          "Aggregates models from OpenAI, Anthropic, Gemini, Cohere, Mistral, DeepSeek, " +
          "xAI, Groq, Cerebras, Perplexity, OpenRouter, vLLM, Ollama, ZhipuAI, Bedrock, and MiniMax.",
        tags: ["LLM Proxy"],
        response: constructResponseSchema(
          z.object({
            object: z.literal("list"),
            data: z.array(
              z.object({
                id: z.string(),
                object: z.literal("model"),
                created: z.number(),
                owned_by: z.string(),
              }),
            ),
          }),
        ),
      },
    },
    async (request, reply) => {
      logger.debug(
        { url: request.url },
        "[UnifiedProxy] Handling list models request",
      );
      // Return a synthesized list of known models in OpenAI format.
      // In a full implementation, this would query each provider's /models endpoint.
      // For now, return a representative list of known model IDs.
      const allModels = [
        // OpenAI
        { id: "gpt-4o", owned_by: "openai" },
        { id: "gpt-4o-mini", owned_by: "openai" },
        { id: "gpt-4-turbo", owned_by: "openai" },
        { id: "o1-preview", owned_by: "openai" },
        { id: "o1-mini", owned_by: "openai" },
        // Anthropic
        { id: "claude-3-5-sonnet-latest", owned_by: "anthropic" },
        { id: "claude-3-opus-latest", owned_by: "anthropic" },
        { id: "claude-3-haiku-latest", owned_by: "anthropic" },
        // Gemini
        { id: "gemini-2.5-pro-latest", owned_by: "google" },
        { id: "gemini-2.0-flash-latest", owned_by: "google" },
        // Cohere
        { id: "command-r-plus", owned_by: "cohere" },
        { id: "command-r", owned_by: "cohere" },
        // Mistral
        { id: "mistral-large-latest", owned_by: "mistral" },
        { id: "mixtral-8x22b-instruct", owned_by: "mistral" },
        // DeepSeek
        { id: "deepseek-chat", owned_by: "deepseek" },
        { id: "deepseek-coder", owned_by: "deepseek" },
        // xAI
        { id: "grok-2", owned_by: "x-ai" },
        { id: "grok-2-mini", owned_by: "x-ai" },
        // Groq
        { id: "llama-3.3-70b-versatile", owned_by: "groq" },
        { id: "mixtral-8x7b-32768", owned_by: "groq" },
        // Cerebras
        { id: "llama-3.3-70b-instruct", owned_by: "cerebras" },
        // Perplexity
        { id: "sonar-pro", owned_by: "perplexity" },
        { id: "sonar", owned_by: "perplexity" },
        // OpenRouter
        { id: "openrouter/auto", owned_by: "openrouter" },
        // DeepSeek
        { id: "deepseek-chat", owned_by: "deepseek" },
        // MiniMax
        { id: "MiniMax-M2.1", owned_by: "minimax" },
        { id: "MiniMax-M2.5", owned_by: "minimax" },
      ];

      return reply.send({
        object: "list",
        data: allModels.map((m) => ({
          ...m,
          object: "model",
          created: Math.floor(Date.now() / 1000),
        })),
      });
    },
  );
};

export default unifiedProxyRoutes;
