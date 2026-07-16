#!/usr/bin/env bash

readonly AUTORESEARCH_RESULTS_TSV_HEADER=$'timestamp\tcommit\tstatus\tprimary_metric\tespresso_tokens_per_second\tcoreml_tokens_per_second\tspeedup_vs_coreml\ttoken_match\ttext_match\tespresso_first_token_ms\tcoreml_first_token_ms\tespresso_median_token_ms\tcoreml_median_token_ms\tespresso_p95_token_ms\tcoreml_p95_token_ms\tespresso_compile_ms\tcoreml_compile_ms\tespresso_compile_retry_count\tespresso_compile_failure_count\tespresso_exact_head_backend\tespresso_cached_bindings_enabled\toutput_dir\tprompt_id\tchange_summary'
readonly AUTORESEARCH_RESULTS_TSV_HEADER_LEGACY=$'timestamp\tcommit\tstatus\tprimary_metric\tespresso_tokens_per_second\tcoreml_tokens_per_second\tspeedup_vs_coreml\ttoken_match\ttext_match\tespresso_first_token_ms\tcoreml_first_token_ms\tespresso_median_token_ms\tcoreml_median_token_ms\tespresso_p95_token_ms\tcoreml_p95_token_ms\tespresso_compile_ms\tcoreml_compile_ms\toutput_dir\tchange_summary'
readonly AUTORESEARCH_RESULTS_TSV_HEADER_LEGACY_WITH_PROMPT_ID=$'timestamp\tcommit\tstatus\tprimary_metric\tespresso_tokens_per_second\tcoreml_tokens_per_second\tspeedup_vs_coreml\ttoken_match\ttext_match\tespresso_first_token_ms\tcoreml_first_token_ms\tespresso_median_token_ms\tcoreml_median_token_ms\tespresso_p95_token_ms\tcoreml_p95_token_ms\tespresso_compile_ms\tcoreml_compile_ms\toutput_dir\tprompt_id\tchange_summary'

write_autoresearch_results_header() {
  printf '%s\n' "$AUTORESEARCH_RESULTS_TSV_HEADER"
}

normalize_autoresearch_results_tsv() {
  local path="$1"
  local header
  header="$(head -n 1 "$path")"

  case "$header" in
    "$AUTORESEARCH_RESULTS_TSV_HEADER")
      return 0
      ;;
    "$AUTORESEARCH_RESULTS_TSV_HEADER_LEGACY")
      awk -F '\t' -v OFS='\t' -v header="$AUTORESEARCH_RESULTS_TSV_HEADER" '
        NR == 1 { print header; next }
        NF == 0 { next }
        { print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,"","","","",$18,"",$19 }
      ' "$path" >"$path.tmp"
      mv "$path.tmp" "$path"
      ;;
    "$AUTORESEARCH_RESULTS_TSV_HEADER_LEGACY_WITH_PROMPT_ID")
      awk -F '\t' -v OFS='\t' -v header="$AUTORESEARCH_RESULTS_TSV_HEADER" '
        NR == 1 { print header; next }
        NF == 0 { next }
        { print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,"","","","",$18,$19,$20 }
      ' "$path" >"$path.tmp"
      mv "$path.tmp" "$path"
      ;;
    *)
      return 0
      ;;
  esac
}
