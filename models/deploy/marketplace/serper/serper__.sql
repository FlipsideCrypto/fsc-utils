-- depends_on: {{ ref('live') }}
{%- set configs = [
    config_serper_udfs,
    ] -%}
{{- ephemeral_deploy_marketplace(configs) -}}
