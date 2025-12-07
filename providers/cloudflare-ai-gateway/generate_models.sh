#!/usr/bin/env bash
#
# Generate model TOML files for Cloudflare AI Gateway
#
# Required environment variables:
#   CLOUDFLARE_API_TOKEN  - Your Cloudflare API token
#   CLOUDFLARE_ACCOUNT_ID - Your Cloudflare account ID
#   CLOUDFLARE_GATEWAY_ID - Your AI Gateway name/ID
#
# Usage:
#   CLOUDFLARE_API_TOKEN=xxx CLOUDFLARE_ACCOUNT_ID=xxx CLOUDFLARE_GATEWAY_ID=xxx ./generate_models.sh
#

set -eo pipefail

# =============================================================================
# CONFIGURATION: Providers and models to include
# =============================================================================

# Providers to include ALL models from (generated with defaults)
INCLUDE_ALL_PROVIDERS="workers-ai replicate"

# Providers to cross-reference from source provider files
CROSS_REFERENCE_PROVIDERS="openai anthropic"

# For cross-referenced providers, only include these well-known models (regex patterns)
# Format: "provider/model-pattern"
# Use $ anchor for exact matches to avoid dated versions and variants
WELL_KNOWN_MODELS=(
  # OpenAI - canonical names only, no dated versions
  "openai/gpt-5.1$"
  "openai/gpt-5.1-codex$"
  "openai/gpt-4o$"
  "openai/gpt-4o-mini$"
  "openai/gpt-4-turbo$"
  "openai/gpt-4$"
  "openai/gpt-3.5-turbo$"
  "openai/o1$"
  "openai/o1-mini$"
  "openai/o1-preview$"
  "openai/o3$"
  "openai/o3-mini$"
  "openai/o3-pro$"
  "openai/o4-mini$"
  
  # Anthropic - canonical names only, no dated versions or duplicates
  "anthropic/claude-sonnet-4.5$"
  "anthropic/claude-opus-4.5$"
  "anthropic/claude-haiku-4.5$"
  "anthropic/claude-opus-4.1$"
  "anthropic/claude-sonnet-4$"
  "anthropic/claude-opus-4$"
  "anthropic/claude-3.5-sonnet$"
  "anthropic/claude-3.5-haiku$"
  "anthropic/claude-3-opus$"
  "anthropic/claude-3-sonnet$"
  "anthropic/claude-3-haiku$"
)

# =============================================================================
# Helper function to get mapped model name for source file lookup
# =============================================================================
get_mapped_name() {
  local model_name="$1"
  case "${model_name}" in
    # Anthropic mappings (Cloudflare uses dots, source uses dashes)
    "claude-sonnet-4.5") echo "claude-sonnet-4-5" ;;
    "claude-opus-4.5") echo "claude-opus-4-5" ;;
    "claude-haiku-4.5") echo "claude-haiku-4-5" ;;
    "claude-opus-4.1") echo "claude-opus-4-1" ;;
    "claude-sonnet-4") echo "claude-sonnet-4-0" ;;
    "claude-opus-4") echo "claude-opus-4-0" ;;
    "claude-3.5-sonnet") echo "claude-3-5-sonnet-20241022" ;;
    "claude-3.5-haiku") echo "claude-3-5-haiku-latest" ;;
    "claude-3-opus") echo "claude-3-opus-20240229" ;;
    "claude-3-sonnet") echo "claude-3-sonnet-20240229" ;;
    "claude-3-haiku") echo "claude-3-haiku-20240307" ;;
    *) echo "${model_name}" ;;
  esac
}

# =============================================================================
# Helper function to check if a model should be included
# =============================================================================
should_include_model() {
  local model_id="$1"
  local provider
  
  # Extract provider from model ID (first path segment)
  provider=$(echo "${model_id}" | cut -d'/' -f1)
  
  # Check if provider is in the "include all" list
  for p in ${INCLUDE_ALL_PROVIDERS}; do
    if [[ "${provider}" == "${p}" ]]; then
      return 0  # Include
    fi
  done
  
  # Check if model matches any well-known pattern
  for pattern in "${WELL_KNOWN_MODELS[@]}"; do
    if echo "${model_id}" | grep -qE "^${pattern}"; then
      return 0  # Include
    fi
  done
  
  return 1  # Exclude
}

# =============================================================================
# Helper function to find source file for cross-referenced models
# =============================================================================
find_source_file() {
  local provider="$1"
  local model_name="$2"
  
  # Check if provider is in cross-reference list
  local is_cross_ref=false
  for p in ${CROSS_REFERENCE_PROVIDERS}; do
    if [[ "${provider}" == "${p}" ]]; then
      is_cross_ref=true
      break
    fi
  done
  
  if [[ "${is_cross_ref}" != "true" ]]; then
    return 1
  fi
  
  # Get mapped name
  local mapped_name
  mapped_name=$(get_mapped_name "${model_name}")
  
  local source_file="${PROVIDERS_DIR}/${provider}/models/${mapped_name}.toml"
  
  if [[ -f "${source_file}" ]]; then
    echo "${source_file}"
    return 0
  fi
  
  # Try original name if mapping didn't work
  if [[ "${mapped_name}" != "${model_name}" ]]; then
    source_file="${PROVIDERS_DIR}/${provider}/models/${model_name}.toml"
    if [[ -f "${source_file}" ]]; then
      echo "${source_file}"
      return 0
    fi
  fi
  
  return 1
}

# =============================================================================
# Main script
# =============================================================================

# Validate required environment variables
if [[ -z "${CLOUDFLARE_API_TOKEN:-}" ]]; then
  echo "Error: CLOUDFLARE_API_TOKEN environment variable is required" >&2
  exit 1
fi

if [[ -z "${CLOUDFLARE_ACCOUNT_ID:-}" ]]; then
  echo "Error: CLOUDFLARE_ACCOUNT_ID environment variable is required" >&2
  exit 1
fi

if [[ -z "${CLOUDFLARE_GATEWAY_ID:-}" ]]; then
  echo "Error: CLOUDFLARE_GATEWAY_ID environment variable is required" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
PROVIDERS_DIR="${SCRIPT_DIR}/../.."

# Fetch models from Cloudflare AI Gateway
echo "Fetching models from Cloudflare AI Gateway..."
API_URL="https://gateway.ai.cloudflare.com/v1/${CLOUDFLARE_ACCOUNT_ID}/${CLOUDFLARE_GATEWAY_ID}/compat/models"

RESPONSE=$(curl -s -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" "${API_URL}")

# Check if the response is valid JSON with data
if ! echo "${RESPONSE}" | jq -e '.data' > /dev/null 2>&1; then
  echo "Error: Invalid API response or no data returned" >&2
  echo "Response: ${RESPONSE}" >&2
  exit 1
fi

MODEL_COUNT=$(echo "${RESPONSE}" | jq '.data | length')

if [[ "${MODEL_COUNT}" -eq 0 ]]; then
  echo "Error: No models found in API response" >&2
  exit 1
fi

echo "Found ${MODEL_COUNT} models from API"

# Create a temporary file to track API model files
API_MODEL_FILES=$(mktemp)
trap "rm -f ${API_MODEL_FILES}" EXIT

INCLUDED_COUNT=0
SKIPPED_COUNT=0
CROSS_REF_COUNT=0

# Process each model from the API response
echo "${RESPONSE}" | jq -c '.data[]' | while IFS= read -r MODEL_JSON; do
  MODEL_ID=$(echo "${MODEL_JSON}" | jq -r '.id')
  COST_IN=$(echo "${MODEL_JSON}" | jq -r '.cost_in // 0')
  COST_OUT=$(echo "${MODEL_JSON}" | jq -r '.cost_out // 0')
  CREATED_AT=$(echo "${MODEL_JSON}" | jq -r '.created_at // 0')
  
  # Skip empty IDs
  [[ -z "${MODEL_ID}" || "${MODEL_ID}" == "null" ]] && continue
  
  # Check if this model should be included
  if ! should_include_model "${MODEL_ID}"; then
    ((SKIPPED_COUNT++)) || true
    continue
  fi
  
  ((INCLUDED_COUNT++)) || true
  
  # Extract provider and model name
  PROVIDER=$(echo "${MODEL_ID}" | cut -d'/' -f1)
  MODEL_NAME=$(echo "${MODEL_ID}" | cut -d'/' -f2-)
  
  # Convert model ID to file path based on the API format:
  # - "workers-ai/@cf/vendor/model-name" -> "workers-ai/model-name.toml"
  # - "anthropic/claude-opus-4-5" -> "anthropic/claude-opus-4-5.toml"
  # - "replicate/meta/llama-3" -> "replicate/meta/llama-3.toml"
  
  if [[ "${MODEL_ID}" == workers-ai/@cf/* ]]; then
    # Workers AI model: workers-ai/@cf/vendor/model-name -> workers-ai/model-name.toml
    MODEL_NAME=$(echo "${MODEL_ID}" | sed 's|workers-ai/@cf/[^/]*/||')
    MODEL_PATH="workers-ai/${MODEL_NAME}.toml"
  else
    # All other models: keep the path structure as-is
    MODEL_PATH="${MODEL_ID}.toml"
  fi
  
  FULL_PATH="${MODELS_DIR}/${MODEL_PATH}"
  echo "${FULL_PATH}" >> "${API_MODEL_FILES}"
  
  # Create directory if needed
  MODEL_DIR=$(dirname "${FULL_PATH}")
  mkdir -p "${MODEL_DIR}"
  
  # Check if we should cross-reference from source provider
  SOURCE_FILE=$(find_source_file "${PROVIDER}" "${MODEL_NAME}" || true)
  
  if [[ -n "${SOURCE_FILE}" && -f "${SOURCE_FILE}" ]]; then
    echo "Cross-referencing: ${MODEL_PATH} <- ${SOURCE_FILE#${PROVIDERS_DIR}/}"
    cp "${SOURCE_FILE}" "${FULL_PATH}"
    ((CROSS_REF_COUNT++)) || true
  else
    # Generate file with defaults for workers-ai, replicate, etc.
    echo "Generating: ${MODEL_PATH}"
    
    # Generate a human-readable name from model ID (use the last part)
    DISPLAY_NAME=$(echo "${MODEL_ID}" | sed 's|.*/||' | sed 's/-/ /g' | sed 's/\b\(.\)/\u\1/g')
    
    # Convert created_at timestamp to date (YYYY-MM-DD)
    if [[ "${CREATED_AT}" != "0" && "${CREATED_AT}" != "null" ]]; then
      RELEASE_DATE=$(date -r "${CREATED_AT}" +%Y-%m-%d 2>/dev/null || date -d "@${CREATED_AT}" +%Y-%m-%d 2>/dev/null || date +%Y-%m-%d)
    else
      RELEASE_DATE=$(date +%Y-%m-%d)
    fi
    
    # Convert cost per token to cost per million tokens
    # API returns cost per token, we need cost per 1M tokens
    # Treat negative or invalid costs as 0
    if [[ "${COST_IN}" != "0" && "${COST_IN}" != "null" ]]; then
      COST_IN_PER_M=$(echo "${COST_IN} * 1000000" | bc -l | sed 's/^\./0./' | sed 's/0*$//' | sed 's/\.$//')
      # If negative, set to 0
      if (( $(echo "${COST_IN_PER_M} < 0" | bc -l) )); then
        COST_IN_PER_M="0"
      fi
    else
      COST_IN_PER_M="0"
    fi
    
    if [[ "${COST_OUT}" != "0" && "${COST_OUT}" != "null" ]]; then
      COST_OUT_PER_M=$(echo "${COST_OUT} * 1000000" | bc -l | sed 's/^\./0./' | sed 's/0*$//' | sed 's/\.$//')
      # If negative, set to 0
      if (( $(echo "${COST_OUT_PER_M} < 0" | bc -l) )); then
        COST_OUT_PER_M="0"
      fi
    else
      COST_OUT_PER_M="0"
    fi
    
    # Always overwrite to ensure data is up to date
    cat > "${FULL_PATH}" << EOF
name = "${DISPLAY_NAME}"
release_date = "${RELEASE_DATE}"
last_updated = "${RELEASE_DATE}"
attachment = false
reasoning = false
temperature = true
tool_call = false
open_weights = false

[cost]
input = ${COST_IN_PER_M}
output = ${COST_OUT_PER_M}

[limit]
context = 128000
output = 16384

[modalities]
input = ["text"]
output = ["text"]
EOF
  fi
done

# Find and remove models that are not in the API response
echo ""
echo "Checking for models to remove..."
REMOVED_COUNT=0

# Find all existing .toml files in models directory
while IFS= read -r -d '' EXISTING_FILE; do
  if ! grep -qxF "${EXISTING_FILE}" "${API_MODEL_FILES}"; then
    REL_PATH="${EXISTING_FILE#${MODELS_DIR}/}"
    echo "Removing model not in API: ${REL_PATH}"
    rm -f "${EXISTING_FILE}"
    ((REMOVED_COUNT++)) || true
  fi
done < <(find "${MODELS_DIR}" -name "*.toml" -type f -print0)

# Clean up empty directories
find "${MODELS_DIR}" -type d -empty -delete 2>/dev/null || true

FINAL_COUNT=$(find "${MODELS_DIR}" -name "*.toml" -type f | wc -l | tr -d ' ')

echo ""
echo "Summary:"
echo "  Models from API: ${MODEL_COUNT}"
echo "  Models included: ${FINAL_COUNT}"
echo "  Models removed: ${REMOVED_COUNT}"
echo ""
echo "Done!"
