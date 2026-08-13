import { jest } from "@jest/globals";

// Regression coverage for the downstream `subfolder` option on tokenizer loading.
// This behaviour predates v4 and was silently dropped by the upstream
// "[v4] Tokenizers.js migration" rewrite, so it is pinned here explicitly.

const mockGetFileMetadata = jest.fn();
jest.unstable_mockModule("../../src/utils/model_registry/get_file_metadata.js", () => ({
  get_file_metadata: mockGetFileMetadata,
}));

const mockGetModelJSON = jest.fn();
jest.unstable_mockModule("../../src/utils/hub.js", () => ({
  getModelJSON: mockGetModelJSON,
}));

// Stand-in for the concrete tokenizer classes, so `AutoTokenizer` can finish
// constructing without a real @huggingface/tokenizers payload.
class StubTokenizer {
  constructor(tokenizerJSON, tokenizerConfig) {
    this.tokenizerJSON = tokenizerJSON;
    this.tokenizerConfig = tokenizerConfig;
  }
}
jest.unstable_mockModule("../../src/models/tokenizers.js", () => ({ BertTokenizer: StubTokenizer }));

const { get_tokenizer_files } = await import("../../src/utils/model_registry/get_tokenizer_files.js");
const { loadTokenizer } = await import("../../src/tokenization_utils.js");
const { AutoTokenizer } = await import("../../src/models/auto/tokenization_auto.js");

const MODEL_ID = "testorg/testmodel";

/** Make the metadata probe report "found" only for the given path. */
const onlyExistsAt = (expectedPath) => mockGetFileMetadata.mockImplementation(async (_modelId, path) => ({ exists: path === expectedPath }));

beforeEach(() => {
  mockGetFileMetadata.mockReset();
  mockGetModelJSON.mockReset();
  mockGetModelJSON.mockImplementation(async (_modelPath, fileName) => ({
    __file: fileName,
    ...(fileName.endsWith("tokenizer_config.json") ? { tokenizer_class: "BertTokenizer" } : {}),
  }));
});

describe("get_tokenizer_files", () => {
  it("probes and returns repo-root paths when no subfolder is given", async () => {
    onlyExistsAt("tokenizer_config.json");

    const files = await get_tokenizer_files(MODEL_ID);

    expect(files).toEqual(["tokenizer.json", "tokenizer_config.json"]);
    expect(mockGetFileMetadata).toHaveBeenCalledWith(MODEL_ID, "tokenizer_config.json", {});
  });

  it("probes inside the subfolder and returns prefixed paths", async () => {
    onlyExistsAt("tokenizer/tokenizer_config.json");

    const files = await get_tokenizer_files(MODEL_ID, { subfolder: "tokenizer" });

    expect(files).toEqual(["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]);
    expect(mockGetFileMetadata).toHaveBeenCalledWith(MODEL_ID, "tokenizer/tokenizer_config.json", {});
  });

  it("reports no tokenizer when the repo only has one inside a subfolder", async () => {
    onlyExistsAt("tokenizer/tokenizer_config.json");

    await expect(get_tokenizer_files(MODEL_ID)).resolves.toEqual([]);
  });

  // An empty subfolder must not turn "tokenizer.json" into "/tokenizer.json":
  // request paths feed the cache key, so a stray slash would miss the cache.
  it.each([null, undefined, ""])("emits no leading slash for subfolder=%p", async (subfolder) => {
    onlyExistsAt("tokenizer_config.json");

    const files = await get_tokenizer_files(MODEL_ID, { subfolder });

    expect(files).toEqual(["tokenizer.json", "tokenizer_config.json"]);
  });
});

describe("loadTokenizer", () => {
  it("requests repo-root files when no subfolder is given", async () => {
    onlyExistsAt("tokenizer_config.json");

    await loadTokenizer(MODEL_ID, {});

    const requested = mockGetModelJSON.mock.calls.map(([, fileName]) => fileName);
    expect(requested).toEqual(["tokenizer.json", "tokenizer_config.json"]);
  });

  it("requests subfolder-prefixed files, without double-prefixing", async () => {
    onlyExistsAt("tokenizer/tokenizer_config.json");

    const [tokenizerJSON, tokenizerConfig] = await loadTokenizer(MODEL_ID, { subfolder: "tokenizer" });

    const requested = mockGetModelJSON.mock.calls.map(([, fileName]) => fileName);
    expect(requested).toEqual(["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]);
    expect(tokenizerJSON.__file).toBe("tokenizer/tokenizer.json");
    expect(tokenizerConfig.__file).toBe("tokenizer/tokenizer_config.json");
  });
});

describe("AutoTokenizer.from_pretrained", () => {
  // The v4 rewrite destructured a fixed whitelist of options, which silently
  // discarded `subfolder` even when callers passed it.
  it("forwards `subfolder` through to the file requests", async () => {
    onlyExistsAt("tokenizer/tokenizer_config.json");

    const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { subfolder: "tokenizer" });

    expect(tokenizer).toBeInstanceOf(StubTokenizer);
    const requested = mockGetModelJSON.mock.calls.map(([, fileName]) => fileName);
    expect(requested).toEqual(["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]);
  });

  it("still loads from the repo root by default", async () => {
    onlyExistsAt("tokenizer_config.json");

    await AutoTokenizer.from_pretrained(MODEL_ID);

    const requested = mockGetModelJSON.mock.calls.map(([, fileName]) => fileName);
    expect(requested).toEqual(["tokenizer.json", "tokenizer_config.json"]);
  });
});
