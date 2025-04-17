/**
 * tinygrad-adapter.ts
 * 
 * This module serves as an adapter between codex and the exo tinygrad interface.
 * It provides functionality to load and execute models using the tinygrad backend.
 */

import { spawn } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

interface ModelConfig {
  modelPath: string;
  engine: string;
  temperature?: number;
  maxTokens?: number;
}

class TinygradAdapter {
  private modelPath: string;
  private engine: string;
  private config: Record<string, any>;
  private pythonProcess: any;
  private temperature: number;
  private maxTokens: number;

  constructor(config: ModelConfig) {
    this.modelPath = config.modelPath;
    this.engine = config.engine || 'TinygradDynamicShardInferenceEngine';
    this.temperature = config.temperature || 0.7;
    this.maxTokens = config.maxTokens || 1024;
    this.config = {};
  }

  /**
   * Initialize the tinygrad adapter by starting the Python process
   */
  async initialize(): Promise<void> {
    // Determine the absolute path to the exo directory
    const rootDir = path.resolve(process.cwd());
    const exoDir = path.join(rootDir, '..', '..', 'exo');
    
    if (!fs.existsSync(exoDir)) {
      throw new Error(`Exo directory not found at ${exoDir}`);
    }

    // Start the Python process with the tinygrad interface
    this.pythonProcess = spawn('python', [
      '-m', 'exo.main',
      '--model', this.modelPath,
      '--engine', this.engine,
      '--temperature', String(this.temperature),
      '--max_tokens', String(this.maxTokens),
      '--server_mode'
    ], {
      cwd: exoDir,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    // Handle process output
    this.pythonProcess.stdout.on('data', (data: Buffer) => {
      const output = data.toString().trim();
      if (output.startsWith('{') && output.endsWith('}')) {
        try {
          this.config = JSON.parse(output);
        } catch (e) {
          console.error('Failed to parse model configuration:', e);
        }
      }
    });

    this.pythonProcess.stderr.on('data', (data: Buffer) => {
      console.error(`[TinygradAdapter] ${data.toString().trim()}`);
    });

    // Wait for initialization
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Timeout waiting for tinygrad initialization'));
      }, 30000);
      
      const checkInterval = setInterval(() => {
        if (Object.keys(this.config).length > 0) {
          clearTimeout(timeout);
          clearInterval(checkInterval);
          resolve();
        }
      }, 100);
    });
  }

  /**
   * Generate text using the loaded model
   */
  async generateText(prompt: string, options: Record<string, any> = {}): Promise<string> {
    if (!this.pythonProcess) {
      throw new Error('Tinygrad adapter not initialized');
    }

    const requestData = JSON.stringify({
      prompt,
      temperature: options.temperature || this.temperature,
      max_tokens: options.maxTokens || this.maxTokens,
      ...options
    });

    // Send the request to the Python process
    this.pythonProcess.stdin.write(requestData + '\n');

    // Wait for the response
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Timeout waiting for model response'));
      }, 60000);
      
      const onData = (data: Buffer) => {
        const response = data.toString().trim();
        if (response.startsWith('RESPONSE:')) {
          const jsonStr = response.substring('RESPONSE:'.length).trim();
          try {
            const result = JSON.parse(jsonStr);
            clearTimeout(timeout);
            this.pythonProcess.stdout.removeListener('data', onData);
            resolve(result.text);
          } catch (e) {
            reject(new Error(`Failed to parse response: ${e}`));
          }
        }
      };
      
      this.pythonProcess.stdout.on('data', onData);
    });
  }

  /**
   * Clean up resources
   */
  async shutdown(): Promise<void> {
    if (this.pythonProcess) {
      this.pythonProcess.stdin.write('EXIT\n');
      return new Promise((resolve) => {
        this.pythonProcess.on('close', () => {
          this.pythonProcess = null;
          resolve();
        });
        
        // Force kill after timeout
        setTimeout(() => {
          if (this.pythonProcess) {
            this.pythonProcess.kill();
            this.pythonProcess = null;
            resolve();
          }
        }, 5000);
      });
    }
  }
}

export { TinygradAdapter, ModelConfig }; 