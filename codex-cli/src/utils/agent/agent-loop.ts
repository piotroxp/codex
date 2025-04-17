import type { ReviewDecision } from "./review.js";
import type { ApplyPatchCommand, ApprovalPolicy } from "../../approvals.js";
import type { AppConfig } from "../config.js";
import type {
  ResponseInputItem,
  ResponseItem,
  ResponseInputMessageContent,
  ResponseOutputText,
} from "openai/resources/responses/responses.mjs";

import { log, isLoggingEnabled } from "./log.js";
import {
  getSessionId,
  setCurrentModel,
  setSessionId,
} from "../session.js";
import { randomUUID } from "node:crypto";

// Define the type for the completion function
export type CompletionFunction = (prompt: string) => Promise<string>;

export type CommandConfirmation = {
  review: ReviewDecision;
  applyPatch?: ApplyPatchCommand | undefined;
  customDenyMessage?: string;
};

type AgentLoopParams = {
  model: string;
  config?: AppConfig;
  instructions?: string;
  approvalPolicy: ApprovalPolicy;
  onItem: (item: ResponseItem) => void;
  onLoading: (loading: boolean) => void;
  completionFn: CompletionFunction;

  /** Called when the command is not auto-approved to request explicit user review. */
  getCommandConfirmation: (
    command: Array<string>,
    applyPatch: ApplyPatchCommand | undefined,
  ) => Promise<CommandConfirmation>;
  onLastResponseId: (lastResponseId: string) => void;
};

export class AgentLoop {
  private model: string;
  private instructions?: string;
  private completionFn: CompletionFunction;

  private onItem: (item: ResponseItem) => void;
  private onLoading: (loading: boolean) => void;
  private onLastResponseId: (lastResponseId: string) => void;

  /**
   * A reference to the currently active stream returned from the OpenAI
   * client. We keep this so that we can abort the request if the user decides
   * to interrupt the current task (e.g. via the escape hot‑key).
   */
  private currentStream: unknown | null = null;
  /** Incremented with every call to `run()`. Allows us to ignore stray events
   * from streams that belong to a previous run which might still be emitting
   * after the user has canceled and issued a new command. */
  private generation = 0;
  /** AbortController for in‑progress tool calls (e.g. shell commands). */
  private execAbortController: AbortController | null = null;
  /** Set to true when `cancel()` is called so `run()` can exit early. */
  private canceled = false;
  /** Function calls that were emitted by the model but never answered because
   *  the user cancelled the run.  We keep the `call_id`s around so the *next*
   *  request can send a dummy `function_call_output` that satisfies the
   *  contract and prevents the
   *    400 | No tool output found for function call …
   *  error from OpenAI. */
  private pendingAborts: Set<string> = new Set();
  /** Set to true by `terminate()` – prevents any further use of the instance. */
  private terminated = false;
  /** Master abort controller – fires when terminate() is invoked. */
  private readonly hardAbort = new AbortController();

  public sessionId: string;
  /*
   * Cumulative thinking time across this AgentLoop instance (ms).
   * Currently not used anywhere – comment out to keep the strict compiler
   * happy under `noUnusedLocals`.  Restore when telemetry support lands.
   */
  // private cumulativeThinkingMs = 0;
  constructor({
    model,
    instructions,
    approvalPolicy,
    config,
    onItem,
    onLoading,
    getCommandConfirmation,
    onLastResponseId,
    completionFn,
  }: AgentLoopParams & { config?: AppConfig }) {
    this.model = model;
    this.instructions = instructions;
    this.completionFn = completionFn;

    // If no `config` has been provided we derive a minimal stub so that the
    // rest of the implementation can rely on `this.config` always being a
    // defined object.  We purposefully copy over the `model` and
    // `instructions` that have already been passed explicitly so that
    // downstream consumers (e.g. telemetry) still observe the correct values.
    const effectiveConfig =
      config ??
      ({
        model,
        instructions: instructions ?? "",
      } as AppConfig);
    this.onItem = onItem;
    this.onLoading = onLoading;
    this.getCommandConfirmation = getCommandConfirmation;
    this.onLastResponseId = onLastResponseId;
    this.sessionId = getSessionId() || randomUUID().replaceAll("-", "");

    setSessionId(this.sessionId);
    setCurrentModel(this.model);

    this.hardAbort = new AbortController();

    this.hardAbort.signal.addEventListener(
      "abort",
      () => this.execAbortController?.abort(),
      { once: true },
    );
  }

  /**
   * Abort the ongoing request/stream, if any.
   * TODO: Adapt cancellation for non-streaming completionFn if necessary.
   * For now, it mainly signals cancellation via flags.
   */
  public cancel(): void {
    if (this.terminated) {
      return;
    }

    // Reset the current stream placeholder
    this.currentStream = null;
    if (isLoggingEnabled()) {
      log(
        `AgentLoop.cancel() invoked – currentStream=${Boolean(
          this.currentStream,
        )} execAbortController=${Boolean(
          this.execAbortController,
        )} generation=${this.generation}`,
      );
    }

    this.canceled = true;

    // Abort any in-progress tool calls (though tool calls are being removed from run())
    this.execAbortController?.abort();

    // Create a new abort controller for future tool calls
    this.execAbortController = new AbortController();
    if (isLoggingEnabled()) {
      log("AgentLoop.cancel(): execAbortController.abort() called");
    }

    this.onLoading(false);

    this.generation += 1;
    if (isLoggingEnabled()) {
      log(`AgentLoop.cancel(): generation bumped to ${this.generation}`);
    }
  }

  /**
   * Hard‑stop the agent loop. After calling this method the instance becomes
   * unusable: any in‑flight operations are aborted and subsequent invocations
   * of `run()` will throw.
   */
  public terminate(): void {
    if (this.terminated) {
      return;
    }
    this.terminated = true;

    this.hardAbort.abort();

    this.cancel();
  }

  public async run(
    input: Array<ResponseInputItem>,
  ): Promise<void> {
    try {
      if (this.terminated) {
        throw new Error("AgentLoop has been terminated");
      }
      const thinkingStart = Date.now();
      const thisGeneration = ++this.generation;

      this.canceled = false;
      this.currentStream = null;

      this.execAbortController = new AbortController();
      if (isLoggingEnabled()) {
        log(
          `AgentLoop.run(): new execAbortController created (${this.execAbortController.signal}) for generation ${this.generation}`,
        );
      }

      try {
        this.onLastResponseId("");
      } catch { /* ignore */ }

      this.pendingAborts.clear();

      let fullPrompt = "";
      for (const item of input) {
        this.onItem(item as ResponseItem);

        if (item.type === 'message') {
          const contentArray = Array.isArray(item.content) ? item.content : [];
          const contentText = contentArray
            .map((c: ResponseInputMessageContent) => (c.type === 'input_text' ? c.text : ''))
            .join(' ');
          fullPrompt += `${item.role === 'user' ? 'User' : 'Assistant'}: ${contentText}\n`;
        } else if (item.type === 'function_call_output') {
          fullPrompt += `Function Result (${item.call_id}): ${item.output}\n`;
        }
      }

      const prefix = "You are an AI programming assistant.";
      const mergedInstructions = [prefix, this.instructions]
        .filter(Boolean)
        .join("\n");
      fullPrompt = `${mergedInstructions}\n\n${fullPrompt}Assistant:`;

      if (isLoggingEnabled()) {
        log(`AgentLoop.run() Generation ${thisGeneration} - Prompt: ${fullPrompt.substring(0, 500)}`);
      }

      this.onLoading(true);

      let completionText = "";
      let errorOccurred: Error | null = null;
      try {
        completionText = await this.completionFn(fullPrompt);

        if (this.canceled || this.hardAbort.signal.aborted || thisGeneration !== this.generation) {
          log("AgentLoop.run(): Completion received but run was cancelled. Discarding.");
          this.onLoading(false);
          return;
        }

        if (isLoggingEnabled()) {
          log(`AgentLoop.run() Generation ${thisGeneration} - Completion: ${completionText.substring(0, 500)}`);
        }

      } catch (error) {
        log(`AgentLoop.run(): Error during completionFn call: ${error}`);
        errorOccurred = error instanceof Error ? error : new Error(String(error));
      } finally {
        this.onLoading(false);
      }

      if (errorOccurred) {
        const errorItem: ResponseItem = {
          id: `error-${Date.now()}`,
          type: "message",
          role: "system",
          content: [{ type: "input_text", text: `Error during model completion: ${errorOccurred.message}` }],
        };
        this.onItem(errorItem);
      } else if (completionText && !this.canceled && !this.hardAbort.signal.aborted && thisGeneration === this.generation) {
        const assistantMessageContent: ResponseOutputText = {
          type: "output_text",
          text: completionText,
          annotations: [],
        };
        const assistantMessage: ResponseItem = {
          id: `asst-${Date.now()}`,
          type: "message",
          role: "assistant",
          content: [assistantMessageContent],
          model: this.model,
          created: Math.floor(Date.now() / 1000),
          response_format: "text",
          processing_time_ms: Date.now() - thinkingStart,
        };
        this.onItem(assistantMessage);
      }

    } catch (error) {
      log(`AgentLoop.run() top-level error: ${error}`);
      this.onLoading(false);
      const errorItem: ResponseItem = {
        id: `syserr-${Date.now()}`,
        type: "message",
        role: "system",
        content: [
          {
            type: "input_text",
            text: `An unexpected error occurred: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
      if (!this.canceled && !this.hardAbort.signal.aborted) {
        this.onItem(errorItem);
      }
    }
  }
}
