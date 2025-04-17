import fetch from "node-fetch"; // Need to install node-fetch: npm install node-fetch

// Base URL for the exo server
// Use bracket notation for process.env
const EXO_SERVER_URL = process.env['EXO_SERVER_URL'] || "http://localhost:8000";

// Keep RECOMMENDED_MODELS if needed for UI/defaults, but it's now just cosmetic
// as only phi-4 is supported via the exo server currently.
export const RECOMMENDED_MODELS: Array<string> = ["phi-4"]; // Update to reflect current reality

// --- New functions to interact with Exo server --- //

async function callExoApi<T>(endpoint: string, body: unknown): Promise<T> {
  try {
    const response = await fetch(`${EXO_SERVER_URL}${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      console.error(`Exo API Error (${response.status}): ${errorBody}`);
      throw new Error(`Exo API request failed to ${endpoint} with status ${response.status}`);
    }
    return (await response.json()) as T;
  } catch (error) {
    console.error(`Error calling Exo API endpoint ${endpoint}:`, error);
    // Consider more specific error handling or re-throwing
    throw error;
  }
}

export async function encodeText(text: string): Promise<number[]> {
  const response = await callExoApi<{ tokens: number[] }>("/encode", { text });
  return response.tokens;
}

export async function decodeTokens(tokens: number[]): Promise<string> {
  const response = await callExoApi<{ text: string }>("/decode", { tokens });
  return response.text;
}

// Placeholder for the actual inference logic that uses the /infer endpoint
// This will replace the OpenAI completion calls.
// It needs to handle tokenization, calling /infer, and sampling the logits.
export async function getModelCompletion(
  prompt: string,
  // Add other parameters like max_tokens, temperature, etc. as needed
): Promise<string> {
  console.log(`Getting completion for prompt: ${prompt.substring(0, 100)}...`);

  // 1. Encode the prompt
  const promptTokens = await encodeText(prompt);

  // --- Simple greedy decoding example --- 
  // TODO: Implement more sophisticated sampling (temperature, top-p)
  // TODO: Handle max_tokens limit
  // TODO: Implement streaming if needed
  let generatedTokens: number[] = [];
  const maxGeneratedTokens = 100; // Example limit
  let currentTokens = promptTokens;

  for (let i = 0; i < maxGeneratedTokens; i++) {
      // 2. Call /infer with current sequence
      const inferResponse = await callExoApi<{ logits?: number[][][] }>("/infer", {
          request_id: `codex-${Date.now()}`, // Simple request ID
          tokens: currentTokens,
      });

      // Add null/undefined checks
      if (!inferResponse?.logits || inferResponse.logits.length === 0 || inferResponse.logits[0].length === 0) {
          console.error("Invalid logits received from Exo server:", inferResponse);
          throw new Error("Invalid logits received from Exo server");
      }

      const batchLogits = inferResponse.logits[0];
      const lastTokenLogits = batchLogits[batchLogits.length - 1];

      if (!lastTokenLogits || lastTokenLogits.length === 0) {
        console.error("Invalid last token logits received:", lastTokenLogits);
        throw new Error("Invalid last token logits received");
      }

      // Find index of max logit (greedy sampling)
      let maxLogit = -Infinity;
      let nextToken = -1;
      for (let j=0; j < lastTokenLogits.length; j++) {
          if (lastTokenLogits[j] > maxLogit) {
              maxLogit = lastTokenLogits[j];
              nextToken = j;
          }
      }

      // Basic end-of-sequence check (replace with actual EOS token ID if known)
      // This is a placeholder - need the actual EOS token ID from the tokenizer
      // const EOS_TOKEN_ID = 2; // Example for some models
      // if (nextToken === EOS_TOKEN_ID) { 
      //     break;
      // }

      if (nextToken < 0) {
        console.warn("No valid next token found, stopping generation.");
        break;
      }

      // 5. Add to generated tokens and update current sequence
      generatedTokens.push(nextToken);
      currentTokens = [...currentTokens, nextToken];
  }

  // 6. Decode the generated tokens
  const completionText = await decodeTokens(generatedTokens);
  console.log(`Generated completion: ${completionText.substring(0, 100)}...`);

  return completionText;
}

// --- Remove or comment out old OpenAI functions --- //

/*
import { OPENAI_API_KEY } from "./config";
import OpenAI from "openai";

const MODEL_LIST_TIMEOUT_MS = 2_000; // 2 seconds

let modelsPromise: Promise<Array<string>> | null = null;

async function fetchModels(): Promise<Array<string>> {
  // If the user has not configured an API key we cannot hit the network.
  if (!OPENAI_API_KEY) {
    return RECOMMENDED_MODELS;
  }

  try {
    const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
    const list = await openai.models.list();

    const models: Array<string> = [];
    for await (const model of list as AsyncIterable<{ id?: string }>) {
      if (model && typeof model.id === "string") {
        models.push(model.id);
      }
    }

    return models.sort();
  } catch {
    return [];
  }
}

export function preloadModels(): void {
  // This might not be necessary anymore unless we pre-check exo server status
  console.log("PreloadModels (OpenAI) skipped.")
  // if (!modelsPromise) {
  //   // Fire‑and‑forget – callers that truly need the list should `await`
  //   // `getAvailableModels()` instead.
  //   void getAvailableModels();
  // }
}

export async function getAvailableModels(): Promise<Array<string>> {
  // Return the hardcoded list as we only support phi-4 via exo for now
  console.log("getAvailableModels returning hardcoded list:", RECOMMENDED_MODELS)
  return Promise.resolve(RECOMMENDED_MODELS);
  // if (!modelsPromise) {
  //   modelsPromise = fetchModels();
  // }
  // return modelsPromise;
}

export async function isModelSupportedForResponses(
  model: string | undefined | null,
): Promise<boolean> {
  // Assume phi-4 is always supported if the server is running
  // Could add a health check endpoint to the exo server later
  console.log(`isModelSupportedForResponses checking: ${model}`);
  return model === "phi-4";
  // ... (rest of the old OpenAI logic removed) ...
}
*/

// Add dummy versions of removed functions if they are still called elsewhere
// to avoid breaking changes immediately. Mark them as deprecated.

/** @deprecated Replaced by Exo server check or removed. */
export function preloadModels(): void {
  console.log("preloadModels called (now a no-op).");
  // Optionally ping the exo server health endpoint here if one exists
}

/** @deprecated Replaced by Exo server check or removed. */
export async function getAvailableModels(): Promise<Array<string>> {
  console.log("getAvailableModels called (returns hardcoded ['phi-4']).");
  return Promise.resolve(["phi-4"]);
}

/** @deprecated Replaced by Exo server check or removed. */
export async function isModelSupportedForResponses(
  model: string | undefined | null,
): Promise<boolean> {
  console.log(`isModelSupportedForResponses called for ${model} (returns true if 'phi-4').`);
  // For now, only allow phi-4
  return model === "phi-4";
}
