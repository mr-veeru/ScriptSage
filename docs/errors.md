# ScriptSage Error Catalog

This document lists common errors encountered when using ScriptSage, along with their causes and solutions.

## Error Categories

- [API Errors](#api-errors)
- [Model Errors](#model-errors)
- [Language Detection Errors](#language-detection-errors)
- [AST Analysis Errors](#ast-analysis-errors)
- [Setup Errors](#setup-errors)

## API Errors

### Error 400: Bad Request

**Error Message**: `Invalid request format`

**Cause**: The JSON payload sent to the API is malformed or missing required fields.

**Solution**: Ensure your request includes all required fields (`code` for `/api/analyze`). Validate your JSON before sending.

### Error 413: Payload Too Large

**Error Message**: `Code snippet exceeds maximum size`

**Cause**: The submitted code exceeds the size limit (default: 500KB).

**Solution**: Submit smaller code snippets or increase the `MAX_CODE_SIZE` in your `.env` file.

### Error 429: Too Many Requests

**Error Message**: `Rate limit exceeded`

**Cause**: Too many requests sent from the same IP address.

**Solution**: Implement backoff strategy in your client or request an API key for higher limits.

## Model Errors

### Model Not Found

**Error Message**: `Model file not found: <model_name>`

**Cause**: The required ML model file is missing from the `models` directory.

**Solution**: 
1. Ensure you've run `quickstart.py` with the `--offline` flag to use bundled models
2. Manually download models from the release page
3. Train models locally using `python backend/core/model_trainer.py`

### Model Version Mismatch

**Error Message**: `Model version incompatible: expected <version>, got <version>`

**Cause**: The model was trained with a different version of ScriptSage.

**Solution**: Retrain models with your current version or download compatible models.

## Language Detection Errors

### Unknown Language

**Error Message**: `Could not determine language with sufficient confidence`

**Cause**: The code snippet doesn't match known patterns or is too short.

**Solution**: 
1. Ensure the code is valid and contains recognizable syntax
2. Lower the confidence threshold in `.env` (e.g., `LANGUAGE_DETECTION_THRESHOLD=0.5`)
3. Provide more context or a larger snippet

### Multiple Language Match

**Error Message**: `Multiple language matches with similar confidence`

**Cause**: The code could belong to multiple similar languages.

**Solution**:
1. Include language-specific keywords or syntax in your code
2. Specify the language manually in the request: `{"code": "...", "language_hint": "Python"}`

## AST Analysis Errors

### Parser Error

**Error Message**: `Failed to parse <language> code: <error details>`

**Cause**: The code contains syntax errors or unsupported language features.

**Solution**:
1. Validate your code syntax
2. For Python, ensure compatibility with the AST module version being used
3. For other languages, only basic analysis is available

### Memory Error During Parsing

**Error Message**: `Memory error during AST generation`

**Cause**: The code is too complex or large for AST analysis.

**Solution**:
1. Submit smaller code snippets
2. Set `DISABLE_AST_ANALYSIS=True` in `.env` for large files
3. Increase server memory limits

## Setup Errors

### GitHub API Rate Limit

**Error Message**: `GitHub API rate limit exceeded during data collection`

**Cause**: Too many requests to GitHub API without authentication.

**Solution**:
1. Add your GitHub token to `.env`: `GITHUB_API_TOKEN=your_token`
2. Run with the `--offline` flag to use bundled models
3. Wait for rate limit reset (usually 1 hour)

### Database Initialization Failure

**Error Message**: `Failed to initialize SQLite database`

**Cause**: Insufficient permissions or disk space.

**Solution**:
1. Ensure write permissions to the `backend/data` directory
2. Check available disk space
3. Verify SQLite installation

### Missing Frontend Dependencies

**Error Message**: `Failed to resolve npm dependencies`

**Cause**: Network issues or incompatible npm/Node.js versions.

**Solution**:
1. Update Node.js and npm to latest versions
2. Run `npm cache clean --force` before installation
3. Check network connection

## Graceful Degradation

ScriptSage implements graceful degradation to handle component failures:

1. If AST analysis fails, falls back to basic analysis
2. If ML model fails, uses rule-based fallback
3. If a language isn't recognized, attempts generic analysis

## Reporting New Errors

If you encounter an error not listed here, please report it by:

1. Creating an issue in the GitHub repository
2. Including the full error message and stack trace
3. Describing the steps to reproduce
4. Attaching a sample code snippet (if possible) 