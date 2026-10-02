import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { AutoProcessor } from "../../src/models/auto/processing_auto.js";
import { env } from "../../src/env.js";

let modelPath;
let previousAllowRemoteModels;

beforeAll(() => {
  previousAllowRemoteModels = env.allowRemoteModels;
  env.allowRemoteModels = false;
  modelPath = fs.mkdtempSync(path.join(os.tmpdir(), "transformers-custom-processor-"));
  fs.writeFileSync(path.join(modelPath, "preprocessor_config.json"), JSON.stringify({ processor_class: "DownstreamProcessor" }));
});

afterAll(() => {
  delete AutoProcessor.PROCESSOR_CLASS_MAPPING.DownstreamProcessor;
  env.allowRemoteModels = previousAllowRemoteModels;
  fs.rmSync(modelPath, { recursive: true, force: true });
});

describe("downstream processor registration", () => {
  it("resolves a registered processor and forwards loading options", async () => {
    const options = { local_files_only: true };
    const calls = [];
    const processor = { custom: true };
    AutoProcessor.PROCESSOR_CLASS_MAPPING.DownstreamProcessor = class {
      static async from_pretrained(id, receivedOptions) {
        calls.push([id, receivedOptions]);
        return processor;
      }
    };

    expect(await AutoProcessor.from_pretrained(modelPath, options)).toBe(processor);
    expect(calls).toEqual([[modelPath, options]]);
  });

  it("allows a subclass to override registration without changing the base mapping", async () => {
    const registered = AutoProcessor.PROCESSOR_CLASS_MAPPING.DownstreamProcessor;
    const processor = { subclass: true };
    class CustomAutoProcessor extends AutoProcessor {
      static PROCESSOR_CLASS_MAPPING = {
        ...AutoProcessor.PROCESSOR_CLASS_MAPPING,
        DownstreamProcessor: class {
          static async from_pretrained() {
            return processor;
          }
        },
      };
    }

    expect(await CustomAutoProcessor.from_pretrained(modelPath, { local_files_only: true })).toBe(processor);
    expect(AutoProcessor.PROCESSOR_CLASS_MAPPING.DownstreamProcessor).toBe(registered);
  });
});
