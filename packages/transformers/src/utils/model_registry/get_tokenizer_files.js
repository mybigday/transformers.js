import { get_file_metadata } from './get_file_metadata.js';

/**
 * Returns the list of files that will be loaded for a tokenizer.
 * Automatically detects whether the model has tokenizer files.
 *
 * @param {string} modelId The model id to check for tokenizer files
 * @param {Object} [options] Optional parameters
 * @param {string|null} [options.subfolder=null] In case the tokenizer files are located inside a subfolder
 * of the model repo, you can specify the folder name here. Returned paths are prefixed with it.
 * @returns {Promise<string[]>} An array of file names that will be loaded
 */
export async function get_tokenizer_files(modelId, { subfolder = null } = {}) {
    if (!modelId) {
        throw new Error('modelId is required for get_tokenizer_files');
    }

    const prefix = subfolder ? `${subfolder}/` : '';

    const metadata = await get_file_metadata(modelId, `${prefix}tokenizer_config.json`, {});
    if (metadata.exists) {
        return [`${prefix}tokenizer.json`, `${prefix}tokenizer_config.json`];
    }

    return [];
}
