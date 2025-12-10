{% macro python_hex_to_int() %}
def hex_to_int(hex) -> str:
    """
    Converts hex (of any size) to int (as a string). Snowflake and java script can only handle up to 64-bit (38 digits of precision)
    hex_to_int('200000000000000000000000000000211');
    >> 680564733841876926926749214863536423441
    hex_to_int('0x200000000000000000000000000000211');
    >> 680564733841876926926749214863536423441
    hex_to_int(NULL);
    >> NULL
    """
    return (str(int(hex, 16)) if hex and hex != "0x" else None)
{% endmacro %}


{% macro python_udf_hex_to_int_with_encoding() %}
def hex_to_int(encoding, hex) -> str:
  """
  Converts hex (of any size) to int (as a string). Snowflake and java script can only handle up to 64-bit (38 digits of precision)
  hex_to_int('hex', '200000000000000000000000000000211');
  >> 680564733841876926926749214863536423441
  hex_to_int('hex', '0x200000000000000000000000000000211');
  >> 680564733841876926926749214863536423441
  hex_to_int('hex', NULL);
  >> NULL
  hex_to_int('s2c', 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffe5b83acf');
  >> -440911153
  """
  if not hex:
    return None
  if encoding.lower() == 's2c':
    if hex[0:2].lower() != '0x':
      hex = f'0x{hex}'

    bits = len(hex[2:])*4
    value = int(hex, 0)
    if value & (1 << (bits-1)):
        value -= 1 << bits
    return str(value)
  else:
    return str(int(hex, 16))
{% endmacro %}

{% macro create_udf_keccak256() %}
from Crypto.Hash import keccak

def udf_encode(event_name):
    keccak_hash = keccak.new(digest_bits=256)
    keccak_hash.update(event_name.encode('utf-8'))
    return '0x' + keccak_hash.hexdigest()
{% endmacro %}

{% macro create_udf_evm_text_signature() %}

def get_simplified_signature(abi):
    def generate_signature(inputs):
        signature_parts = []
        for input_data in inputs:
            if 'components' in input_data:
                component_signature_parts = []
                components = input_data['components']
                component_signature_parts.extend(generate_signature(components))
                component_signature_parts[-1] = component_signature_parts[-1].rstrip(",")
                if input_data['type'].endswith('[]'):
                    signature_parts.append("(" + "".join(component_signature_parts) + ")[],")
                else:
                    signature_parts.append("(" + "".join(component_signature_parts) + "),")
            else:
                signature_parts.append(input_data['type'].replace('enum ', '').replace(' payable', '') + ",")
        return signature_parts

    signature_parts = [abi['name'] + "("]
    signature_parts.extend(generate_signature(abi['inputs']))
    signature_parts[-1] = signature_parts[-1].rstrip(",") + ")"
    return "".join(signature_parts)
{% endmacro %}

{% macro create_udf_decimal_adjust() %}

from decimal import Decimal, ROUND_DOWN

def custom_divide(input, adjustment):
    try:
        if adjustment is None or input is None:
            return None

        # Perform the division using Decimal type
        result = Decimal(input) / pow(10, Decimal(adjustment))

        # Determine the number of decimal places in the result
        decimal_places = max(0, -result.as_tuple().exponent)

        # Convert the result to a string representation without scientific notation and with dynamic decimal precision
        result_str = "{:.{prec}f}".format(result, prec=decimal_places)

        return result_str
    except Exception as e:
        return None
{% endmacro %}

{% macro create_udf_cron_to_prior_timestamps() %}
import croniter
import datetime

class TimestampGenerator:

    def __init__(self):
        pass

    def process(self, workflow_name, workflow_schedule):
        for timestamp in self.generate_timestamps(workflow_name, workflow_schedule):
            yield (workflow_name, workflow_schedule, timestamp)

    def generate_timestamps(self, workflow_name, workflow_schedule):
        # Create a cron iterator object
        cron = croniter.croniter(workflow_schedule)

        # Generate timestamps for the prev 10 runs
        timestamps = []
        for i in range(10):
            prev_run = cron.get_prev(datetime.datetime)
            timestamps.append(prev_run)

        return timestamps
{% endmacro %}

{% macro create_udf_transform_logs() %}

from copy import deepcopy

def transform_tuple(components: list, values: list):
    transformed_values = []
    for i, component in enumerate(components):
        if i < len(values):
            if component["type"] == "tuple":
                transformed_values.append({"value": transform_tuple(component["components"], values[i]), **component})
            elif component["type"] == "tuple[]":
                if not values[i]:
                    transformed_values.append({"value": [], **component})
                    continue
                sub_values = [transform_tuple(component["components"], v) for v in values[i]]
                transformed_values.append({"value": sub_values, **component})
            else:
                transformed_values.append({"value": values[i], **component})
    return {item["name"]: item["value"] for item in transformed_values}

def transform_event(event: dict):
    new_event = deepcopy(event)
    if new_event.get("components"):
        components = new_event.get("components")

        if not new_event["value"]:
            return new_event

        if isinstance(new_event["value"][0], list):
            result_list = []
            for value_set in new_event["value"]:
                result_list.append(transform_tuple(components, value_set))
            new_event["value"] = result_list

        else:
            new_event["value"] = transform_tuple(components, new_event["value"])

        return new_event

    else:
        return event

def transform(events: dict):
    try:
        results = [
            transform_event(event) if event.get("decoded") else event
            for event in events["data"]
        ]
        events["data"] = results
        return events
    except:
        return events

{% endmacro %}

{% macro create_udf_base58_to_hex() %}

def transform_base58_to_hex(base58):
    if base58 is None:
        return 'Invalid input'

    ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    base_count = len(ALPHABET)

    num = 0
    leading_zeros = 0

    for char in base58:
        if char == '1':
            leading_zeros += 1
        else:
            break

    for char in base58:
        num *= base_count
        if char in ALPHABET:
            num += ALPHABET.index(char)
        else:
            return 'Invalid character in input'

    hex_string = hex(num)[2:]

    if len(hex_string) % 2 != 0:
        hex_string = '0' + hex_string

    hex_leading_zeros = '00' * leading_zeros

    return '0x' + hex_leading_zeros + hex_string

{% endmacro %}

{% macro create_udf_hex_to_base58() %}

def transform_hex_to_base58(input):
    if input is None or not input.startswith('0x'):
        return 'Invalid input'

    input = input[2:]

    ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    byte_array = bytes.fromhex(input)
    num = int.from_bytes(byte_array, 'big')

    encoded = ''
    while num > 0:
        num, remainder = divmod(num, 58)
        encoded = ALPHABET[remainder] + encoded

    for byte in byte_array:
        if byte == 0:
            encoded = '1' + encoded
        else:
            break

    return encoded

{% endmacro %}

{% macro create_udf_hex_to_bech32() %}

def transform_hex_to_bech32(input, hrp=''):
    CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"

    def bech32_polymod(values):
        generator = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
        checksum = 1
        for value in values:
            top = checksum >> 25
            checksum = ((checksum & 0x1ffffff) << 5) ^ value
            for i in range(5):
                checksum ^= generator[i] if ((top >> i) & 1) else 0
        return checksum

    def bech32_hrp_expand(hrp):
        return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]

    def bech32_create_checksum(hrp, data):
        values = bech32_hrp_expand(hrp) + data
        polymod = bech32_polymod(values + [0, 0, 0, 0, 0, 0]) ^ 1
        return [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]

    def bech32_convertbits(data, from_bits, to_bits, pad=True):
        acc = 0
        bits = 0
        ret = []
        maxv = (1 << to_bits) - 1
        max_acc = (1 << (from_bits + to_bits - 1)) - 1
        for value in data:
            acc = ((acc << from_bits) | value) & max_acc
            bits += from_bits
            while bits >= to_bits:
                bits -= to_bits
                ret.append((acc >> bits) & maxv)
        if pad and bits:
            ret.append((acc << (to_bits - bits)) & maxv)
        return ret

    if input is None or not input.startswith('0x'):
        return 'Invalid input'

    input = input[2:]

    data = bytes.fromhex(input)
    data5bit = bech32_convertbits(list(data), 8, 5)

    if data5bit is None:
        return 'Data conversion failed'

    checksum = bech32_create_checksum(hrp, data5bit)

    return hrp + '1' + ''.join([CHARSET[d] for d in data5bit + checksum])

{% endmacro %}

{% macro create_udf_hex_to_algorand() %}

import hashlib
import base64

def transform_hex_to_algorand(input):
    if input is None or not input.startswith('0x'):
        return 'Invalid input'

    input = input[2:]
    public_key_bytes = bytearray.fromhex(input)

    sha512_256_hash = hashlib.new('sha512_256', public_key_bytes).digest()

    checksum = sha512_256_hash[-4:]

    algorand_address = base64.b32encode(public_key_bytes + checksum).decode('utf-8').rstrip('=')

    return algorand_address

{% endmacro %}

{% macro create_udf_hex_to_tezos() %}

import hashlib

def transform_hex_to_tezos(input, prefix):
    if input is None or not input.startswith('0x'):
        return 'Invalid input'

    input = input[2:]

    if len(input) != 40:
        return 'Invalid length'

    hash_bytes = bytes.fromhex(input)

    prefixes = {
        'tz1': '06a19f',  # Ed25519
        'tz2': '06a1a1',  # Secp256k1
        'tz3': '06a1a4'   # P-256
    }

    if prefix not in prefixes:
        return 'Invalid prefix: Must be tz1, tz2, or tz3'

    prefix_bytes = bytes.fromhex(prefixes[prefix])

    prefixed_hash = prefix_bytes + hash_bytes

    checksum = hashlib.sha256(hashlib.sha256(prefixed_hash).digest()).digest()[:4]

    full_hash = prefixed_hash + checksum

    tezos_address = transform_hex_to_base58(full_hash.hex())

    return tezos_address

def transform_hex_to_base58(input):
    if input is None:
        return None

    ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    byte_array = bytes.fromhex(input)
    num = int.from_bytes(byte_array, 'big')

    encoded = ''
    while num > 0:
        num, remainder = divmod(num, 58)
        encoded = ALPHABET[remainder] + encoded

    for byte in byte_array:
        if byte == 0:
            encoded = '1' + encoded
        else:
            break

    return encoded

{% endmacro %}

{% macro create_udf_detect_overflowed_responses() %}

import pandas as pd
from snowflake.snowpark.files import SnowflakeFile

VARCHAR_MAX = 16_777_216
def main(file_url, index_cols):
    with SnowflakeFile.open(file_url, 'rb') as f:
        df = pd.read_json(f, lines=True, compression='gzip')
    data_length = df["data"].astype(str).apply(len)
    return df[data_length > VARCHAR_MAX][index_cols].values.tolist()

{% endmacro %}

{% macro create_udtf_flatten_overflowed_responses() %}

import logging
import simplejson as json

import numpy as np
import pandas as pd
from snowflake.snowpark.files import SnowflakeFile

VARCHAR_MAX = 16_777_216

logger = logging.getLogger("udtf_flatten_overflowed_responses")

class Flatten:
    """
    Recursive function to flatten a nested JSON file
    """

    def __init__(self, mode: str, exploded_key: list) -> None:
        self.mode = mode
        self.exploded_key = exploded_key

    def _flatten_response(
        self,
        response_key: str,
        responses: str,
        block_number: int,
        metadata: dict,
        seq_index: int = 0,
        path: str = "",
    ):
        """
        Example:

        input: {"a":1, "b":[77,88], "c": {"d":"X"}}

        output:
        - SEQ: A unique sequence number associated with the input record; the sequence is not guaranteed to be gap-free or ordered in any particular way.
        - KEY: For maps or objects, this column contains the key to the exploded value.
        - PATH: The path to the element within a data structure which needs to be flattened.
        - INDEX: The index of the element, if it is an array; otherwise NULL.
        - VALUE_: The value of the element of the flattened array/object.

        """
        exploded_data = []
        if self.mode == "array":
            check_mode = isinstance(responses, list)
        elif self.mode == "dict":
            check_mode = isinstance(responses, dict)
        elif self.mode == "both":
            check_mode = isinstance(responses, list) or isinstance(responses, dict)

        if check_mode:
            if isinstance(responses, dict):
                looped_keys = responses.keys()
                for key in looped_keys:
                    next_path = f"{path}.{key}" if path else key
                    index = None
                    exploded_data.append(
                        {
                            "block_number": block_number,
                            "metadata": metadata,
                            "seq": seq_index,
                            "key": key,
                            "path": next_path,
                            "index": index,
                            "value_": responses[key],
                        }
                    )
                    exploded_data.extend(
                        self._flatten_response(
                            key,
                            responses[key],
                            block_number,
                            metadata,
                            seq_index,
                            next_path,
                        )
                    )

            elif isinstance(responses, list):
                looped_keys = range(len(responses))
                if response_key in self.exploded_key or len(self.exploded_key) == 0:
                    for item_i, item in enumerate(responses):
                        if response_key == "result":
                            seq_index += 1
                        index = item_i
                        exploded_data.append(
                            {
                                "block_number": block_number,
                                "metadata": metadata,
                                "seq": seq_index,
                                "key": None,
                                "path": f"{path}[{item_i}]",
                                "index": index,
                                "value_": item,
                            }
                        )
                        exploded_data.extend(
                            self._flatten_response(
                                item_i,
                                item,
                                block_number,
                                metadata,
                                seq_index,
                                f"{path}[{item_i}]",
                            )
                        )

        return exploded_data

class FlattenRows:
    """
    Recursive function to flatten a given JSON file from Snowflake stage
    """
    def process(self, file_url: str, index_cols: list, index_vals: list):
        with SnowflakeFile.open(file_url, 'rb') as f:
            df = pd.read_json(f, lines=True, compression='gzip')

        df.set_index(index_cols, inplace=True, drop=False)
        df = df.loc[index_vals]

        flattener = Flatten(mode="both", exploded_key=[])

        df["value_"] = df.apply(
                lambda x: flattener._flatten_response(
                    block_number=x["block_number"], metadata=x["metadata"], responses=x["data"], response_key=None
                ),
                axis="columns",
            )
        df["value_"] = df["value_"].apply(pd.DataFrame.from_records)
        df["index_cols"] = df.index
        df = df[["index_cols", "value_"]]
        flattened = pd.concat(
            df["value_"].values.tolist(), keys=df["index_cols"].values.tolist()
        ).droplevel(-1)

        cleansed = flattened.replace({np.nan: None})

        overflow = cleansed["value_"].astype(str).apply(len) > VARCHAR_MAX

        cleansed.loc[overflow, ["value_"]] = None
        temp_index_cols = list(range(len(index_cols)))
        cleansed = cleansed.reset_index(names=temp_index_cols, drop=False)
        cleansed["index_cols"] = cleansed[temp_index_cols].apply(list, axis=1)
        cleansed.drop(columns=temp_index_cols, inplace=True, errors="ignore")
        return list(cleansed[np.roll(cleansed.columns.values, 1).tolist()].itertuples(index=False, name=None))
{% endmacro %}

{% macro create_udf_decompress_zlib() %}
import zlib
import codecs

def decompress_zlib(compressed_string):
    try:
        if not compressed_string:
            return None

        # Remove b prefix and suffix if present
        if compressed_string.startswith("b'") and compressed_string.endswith("'"):
            compressed_string = compressed_string[2:-1]
        elif compressed_string.startswith('b"') and compressed_string.endswith('"'):
            compressed_string = compressed_string[2:-1]

        # Decode the escaped string to bytes
        compressed_bytes = codecs.decode(compressed_string, 'unicode_escape')

        # Convert to bytes if string
        if isinstance(compressed_bytes, str):
            compressed_bytes = compressed_bytes.encode('latin-1')

        # Decompress the zlib data
        decompressed = zlib.decompress(compressed_bytes)

        # Return as UTF-8 string
        return decompressed.decode('utf-8')
    except Exception as e:
        return f"Error decompressing: {str(e)}"
{% endmacro %}

{% macro create_udf_stablecoin_data_parse() %}
import re

class udf_stablecoin_data_parse:
    def process(self, peggeddata_content):
        """Main parsing function"""
        
        def extract_field_value(obj_text, field_name):
            """Extract field value from object text using regex patterns"""
            
            # Handle different field patterns
            patterns = [
                rf'{field_name}\s*:\s*"([^"]*)"',
                rf"{field_name}\s*:\s*'([^']*)'",
                rf'{field_name}\s*:\s*`([^`]*)`',
                rf'{field_name}\s*:\s*(true|false|null|undefined)',
                rf'{field_name}\s*:\s*([^,}}\n]+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, obj_text, re.IGNORECASE | re.DOTALL)
                if match:
                    value = match.group(1).strip()
                    
                    # Clean up the value
                    value = re.sub(r'[,}}\n]', '', value).strip()
                    
                    if value.lower() in ('null', 'undefined', ''):
                        return None
                        
                    # Handle boolean values
                    if value.lower() == 'true':
                        return True
                    if value.lower() == 'false':
                        return False
                        
                    return value
            
            return None

        def convert_value(value, expected_type):
            """Convert value to appropriate type"""
            if value is None:
                return None
                
            if expected_type == 'BOOLEAN':
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    lower = value.lower()
                    if lower == 'true':
                        return True
                    if lower == 'false':
                        return False
                return None
                
            return str(value) if value is not None else None

        try:
            # Find the main array content - make the regex non-greedy but capture everything
            array_match = re.search(r'export\s+default\s*\[(.*)\];?\s*$', peggeddata_content, re.DOTALL)
            if not array_match:
                raise Exception('Could not find exported array in peggedData content')
                
            array_content = array_match.group(1).strip()
            
            # Use a simpler regex-based approach to split objects
            # Remove comments and clean up the array content first
            # Instead of removing line comments entirely, just remove the // markers but keep the content
            clean_content = re.sub(r'^\s*//\s*', '', array_content, flags=re.MULTILINE)  # Remove // at start of lines
            clean_content = re.sub(r'\n\s*//\s*', '\n', clean_content)  # Remove // from middle of lines
            # Instead of removing block comments entirely, just remove the comment markers but keep the content
            clean_content = re.sub(r'/\*', '', clean_content)  # Remove opening block comment markers
            clean_content = re.sub(r'\*/', '', clean_content)  # Remove closing block comment markers
            
            # Find all objects using regex - look for {...} patterns
            # This is more reliable than manual parsing
            object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.finditer(object_pattern, clean_content, re.DOTALL)
            
            objects = []
            for match in matches:
                obj_text = match.group(0).strip()
                if obj_text and len(obj_text) > 10:  # Filter out small matches
                    objects.append(obj_text)
            
            # If the simple regex didn't work, try a more complex nested approach
            if not objects:
                # More complex regex for nested objects
                nested_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
                nested_matches = re.findall(nested_pattern, clean_content, re.DOTALL)
                objects = [obj.strip() for obj in nested_matches if len(obj.strip()) > 20]
            
            # Still no objects? Try manual parsing with better logic
            if not objects:
                objects = []
                current_object = ''
                brace_count = 0
                in_string = False
                string_char = ''
                i = 0
                
                while i < len(clean_content):
                    char = clean_content[i]
                    
                    # Handle string literals
                    if not in_string and char in ('"', "'", '`'):
                        in_string = True
                        string_char = char
                    elif in_string and char == string_char:
                        # Check if it's escaped
                        if i > 0 and clean_content[i-1] != '\\':
                            in_string = False
                            string_char = ''
                    
                    # Handle braces only when not in string
                    if not in_string:
                        if char == '{':
                            if brace_count == 0:
                                current_object = '{'  # Start new object
                            else:
                                current_object += char
                            brace_count += 1
                        elif char == '}':
                            current_object += char
                            brace_count -= 1
                            if brace_count == 0 and current_object.strip():
                                # Complete object found
                                objects.append(current_object.strip())
                                current_object = ''
                        elif brace_count > 0:
                            current_object += char
                    else:
                        if brace_count > 0:
                            current_object += char
                    
                    i += 1
                
            if not objects:
                # Last resort: try splitting on id: pattern
                id_splits = re.split(r'\n\s*id:\s*["\']', clean_content)
                if len(id_splits) > 1:
                    objects = []
                    for i, part in enumerate(id_splits[1:], 1):  # Skip first empty part
                        # Try to reconstruct the object
                        obj_start = clean_content.find(f'id:', clean_content.find(part))
                        if obj_start > 0:
                            # Look backwards for opening brace
                            brace_start = clean_content.rfind('{', 0, obj_start)
                            if brace_start >= 0:
                                # Look forward for matching closing brace
                                brace_count = 0
                                for j in range(brace_start, len(clean_content)):
                                    if clean_content[j] == '{':
                                        brace_count += 1
                                    elif clean_content[j] == '}':
                                        brace_count -= 1
                                        if brace_count == 0:
                                            obj_text = clean_content[brace_start:j+1].strip()
                                            if len(obj_text) > 20:
                                                objects.append(obj_text)
                                            break
                
            if not objects:
                raise Exception(f'No objects found after all parsing attempts. Sample content: {clean_content[:500]}...')
                
            # Process each object and extract the required fields
            for i, obj_text in enumerate(objects):
                try:
                    data = {
                        'id': extract_field_value(obj_text, 'id'),
                        'name': extract_field_value(obj_text, 'name'),
                        'address': extract_field_value(obj_text, 'address'),
                        'symbol': extract_field_value(obj_text, 'symbol'),
                        'onCoinGecko': extract_field_value(obj_text, 'onCoinGecko'),
                        'gecko_id': extract_field_value(obj_text, 'gecko_id'),
                        'cmcId': extract_field_value(obj_text, 'cmcId'),
                        'pegType': extract_field_value(obj_text, 'pegType'),
                        'pegMechanism': extract_field_value(obj_text, 'pegMechanism'),
                        'priceSource': extract_field_value(obj_text, 'priceSource'),
                        'deadFrom': extract_field_value(obj_text, 'deadFrom'),
                        'delisted': extract_field_value(obj_text, 'delisted'),
                        'deprecated': extract_field_value(obj_text, 'deprecated'),
                        'doublecounted': extract_field_value(obj_text, 'doublecounted')
                    }
                    
                    # Only include objects that have at least id and name
                    if data['id'] and data['name']:
                        yield (
                            convert_value(data['id'], 'STRING'),
                            convert_value(data['name'], 'STRING'),
                            convert_value(data['address'], 'STRING'),
                            convert_value(data['symbol'], 'STRING'),
                            convert_value(data['onCoinGecko'], 'BOOLEAN'),
                            convert_value(data['gecko_id'], 'STRING'),
                            convert_value(data['cmcId'], 'STRING'),
                            convert_value(data['pegType'], 'STRING'),
                            convert_value(data['pegMechanism'], 'STRING'),
                            convert_value(data['priceSource'], 'STRING'),
                            convert_value(data['deadFrom'], 'STRING'),
                            convert_value(data['delisted'], 'BOOLEAN'),
                            convert_value(data['deprecated'], 'BOOLEAN'),
                            convert_value(data['doublecounted'], 'BOOLEAN')
                        )
                        
                except Exception as obj_error:
                    # Skip malformed objects but continue processing
                    continue
                    
        except Exception as error:
            raise Exception(f'Error parsing peggedData content: {str(error)}')
{% endmacro %}

{% macro create_udf_encode_contract_call() %}

def encode_call(function_abi, input_values):
    """
    Encodes EVM contract function calls into ABI-encoded calldata.
    
    This function generates complete calldata (selector + encoded params) that can be
    used directly in eth_call JSON-RPC requests to query contract state.
    """
    import eth_abi
    from eth_hash.auto import keccak
    import json
    
    def get_function_signature(abi):
        """
        Generate function signature using the same logic as utils.udf_evm_text_signature.
        
        Examples:
          balanceOf(address)
          transfer(address,uint256)
          swap((address,address,uint256))
        """
        def generate_signature(inputs):
            signature_parts = []
            for input_data in inputs:
                if 'components' in input_data:
                    # Handle nested tuples
                    component_signature_parts = []
                    components = input_data['components']
                    component_signature_parts.extend(generate_signature(components))
                    component_signature_parts[-1] = component_signature_parts[-1].rstrip(",")
                    if input_data['type'].endswith('[]'):
                        signature_parts.append("(" + "".join(component_signature_parts) + ")[],")
                    else:
                        signature_parts.append("(" + "".join(component_signature_parts) + "),")
                else:
                    # Clean up Solidity-specific modifiers
                    signature_parts.append(input_data['type'].replace('enum ', '').replace(' payable', '') + ",")
            return signature_parts

        signature_parts = [abi['name'] + "("]
        signature_parts.extend(generate_signature(abi.get('inputs', [])))
        if len(signature_parts) > 1:
            signature_parts[-1] = signature_parts[-1].rstrip(",") + ")"
        else:
            signature_parts.append(")")
        return "".join(signature_parts)
    
    def function_selector(abi):
        """Calculate 4-byte function selector using Keccak256 hash."""
        signature = get_function_signature(abi)
        hash_bytes = keccak(signature.encode('utf-8'))
        return hash_bytes[:4].hex(), signature
    
    def get_canonical_type(input_spec):
        """
        Convert ABI input spec to canonical type string for eth_abi encoding.
        
        Handles tuple expansion: tuple -> (address,uint256,bytes)
        """
        param_type = input_spec['type']
        
        if param_type.startswith('tuple'):
            components = input_spec.get('components', [])
            component_types = ','.join([get_canonical_type(comp) for comp in components])
            canonical = f"({component_types})"
            
            # Preserve array suffixes: tuple[] -> (address,uint256)[]
            if param_type.endswith('[]'):
                array_suffix = param_type[5:]  # Everything after 'tuple'
                canonical += array_suffix
            
            return canonical
        
        return param_type
    
    def prepare_value(value, param_type, components=None):
        """
        Convert Snowflake values to Python types suitable for eth_abi encoding.
        
        Handles type coercion and format normalization for all Solidity types.
        """
        # Handle null/None values with sensible defaults
        if value is None:
            if param_type.startswith('uint') or param_type.startswith('int'):
                return 0
            elif param_type == 'address':
                return '0x' + '0' * 40
            elif param_type == 'bool':
                return False
            elif param_type.startswith('bytes'):
                return b''
            else:
                return value
        
        # CRITICAL: Check arrays FIRST (before base types)
        # This prevents bytes[] from matching the bytes check
        if param_type.endswith('[]'):
            base_type = param_type[:-2]
            if not isinstance(value, list):
                return []
            
            # Special handling for tuple arrays
            if base_type == 'tuple' and components:
                return [prepare_tuple(v, components) for v in value]
            else:
                return [prepare_value(v, base_type) for v in value]
        
        # Base type conversions
        if param_type == 'address':
            addr = str(value).lower()
            if not addr.startswith('0x'):
                addr = '0x' + addr
            return addr
        
        if param_type.startswith('uint') or param_type.startswith('int'):
            return int(value)
        
        if param_type == 'bool':
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes')
            return bool(value)
        
        if param_type.startswith('bytes'):
            if isinstance(value, str):
                if value.startswith('0x'):
                    value = value[2:]
                return bytes.fromhex(value)
            return value
        
        if param_type == 'string':
            return str(value)
        
        return value
    
    def prepare_tuple(value, components):
        """
        Recursively prepare tuple values, handling nested structures.
        
        Tuples can contain other tuples, arrays, or tuple arrays.
        """
        if not isinstance(value, (list, tuple)):
            # Support dict-style input (by component name)
            if isinstance(value, dict):
                value = [value.get(comp.get('name', f'field_{i}')) 
                        for i, comp in enumerate(components)]
            else:
                return value
        
        result = []
        for i, comp in enumerate(components):
            if i >= len(value):
                result.append(None)
                continue
                
            comp_type = comp['type']
            val = value[i]
            
            # Handle tuple arrays within tuples
            if comp_type.endswith('[]') and comp_type.startswith('tuple'):
                sub_components = comp.get('components', [])
                result.append(prepare_value(val, comp_type, sub_components))
            elif comp_type.startswith('tuple'):
                # Single tuple (not array)
                sub_components = comp.get('components', [])
                result.append(prepare_tuple(val, sub_components))
            else:
                result.append(prepare_value(val, comp_type))
        
        return tuple(result)
    
    try:
        inputs = function_abi.get('inputs', [])
        
        # Calculate selector using battle-tested signature generation
        selector_hex, signature = function_selector(function_abi)
        
        # Functions with no inputs only need the selector
        if not inputs:
            return '0x' + selector_hex
        
        # Prepare values for encoding
        prepared_values = []
        for i, inp in enumerate(inputs):
            if i >= len(input_values):
                prepared_values.append(None)
                continue
            
            value = input_values[i]
            param_type = inp['type']
            
            # Handle tuple arrays at top level
            if param_type.endswith('[]') and param_type.startswith('tuple'):
                components = inp.get('components', [])
                prepared_values.append(prepare_value(value, param_type, components))
            elif param_type.startswith('tuple'):
                # Single tuple (not array)
                components = inp.get('components', [])
                prepared_values.append(prepare_tuple(value, components))
            else:
                prepared_values.append(prepare_value(value, param_type))
        
        # Get canonical type strings for eth_abi (expands tuples)
        types = [get_canonical_type(inp) for inp in inputs]
        
        # Encode parameters using eth_abi
        encoded_params = eth_abi.encode(types, prepared_values).hex()
        
        # Return complete calldata: selector + encoded params
        return '0x' + selector_hex + encoded_params
        
    except Exception as e:
        # Return structured error for debugging
        import traceback
        return json.dumps({
            'error': str(e),
            'traceback': traceback.format_exc(),
            'function': function_abi.get('name', 'unknown'),
            'signature': signature if 'signature' in locals() else 'not computed',
            'selector': '0x' + selector_hex if 'selector_hex' in locals() else 'not computed',
            'types': types if 'types' in locals() else 'not computed'
        })

{% endmacro %}

{% macro udf_encode_contract_call_comment() %}
Encodes EVM contract function calls into hex calldata format for eth_call RPC requests.

PURPOSE:
  Converts human-readable function parameters into ABI-encoded calldata that can be sent
  to Ethereum nodes via JSON-RPC. Handles all Solidity types including complex nested
  structures like tuples and arrays.

PARAMETERS:
  function_abi (VARIANT): 
    - JSON object containing the function ABI definition
    - Must include: "name" (string) and "inputs" (array of input definitions)
    - Each input needs: "name", "type", and optionally "components" for tuples
    
  input_values (ARRAY):
    - Array of values matching the function inputs in order
    - Values should be provided as native Snowflake types:
      * addresses: strings (with or without 0x prefix)
      * uint/int: numbers
      * bool: booleans
      * bytes/bytes32: hex strings (with or without 0x prefix)
      * arrays: Snowflake arrays
      * tuples: Snowflake arrays in component order

RETURNS:
  STRING: Complete calldata as hex string with 0x prefix
    - Format: 0x{4-byte selector}{encoded parameters}
    - Can be used directly in eth_call RPC requests
    - Returns JSON error object if encoding fails

EXAMPLES:

  -- Simple function with no inputs
  SELECT utils.udf_encode_contract_call(
    PARSE_JSON(''{"name": "totalSupply", "inputs": []}''),
    ARRAY_CONSTRUCT()
  );
  -- Returns: 0x18160ddd

  -- Function with single address parameter
  SELECT utils.udf_encode_contract_call(
    PARSE_JSON(''{
      "name": "balanceOf",
      "inputs": [{"name": "account", "type": "address"}]
    }''),
    ARRAY_CONSTRUCT(''0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48'')
  );
  -- Returns: 0x70a08231000000000000000000000000a0b86991c6218b36c1d19d4a2e9eb0ce3606eb48

  -- Function with multiple parameters
  SELECT utils.udf_encode_contract_call(
    PARSE_JSON(''{
      "name": "transfer",
      "inputs": [
        {"name": "to", "type": "address"},
        {"name": "amount", "type": "uint256"}
      ]
    }''),
    ARRAY_CONSTRUCT(''0x1234567890123456789012345678901234567890'', 1000000)
  );

  -- Complex function with nested tuples
  SELECT utils.udf_encode_contract_call(
    PARSE_JSON(''{
      "name": "swap",
      "inputs": [{
        "name": "params",
        "type": "tuple",
        "components": [
          {"name": "tokenIn", "type": "address"},
          {"name": "tokenOut", "type": "address"},
          {"name": "amountIn", "type": "uint256"}
        ]
      }]
    }''),
    ARRAY_CONSTRUCT(
      ARRAY_CONSTRUCT(
        ''0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48'',
        ''0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2'',
        1000000
      )
    )
  );

TYPICAL WORKFLOW:
  1. Get function ABI from crosschain.evm.dim_contract_abis
  2. Prepare input values as Snowflake arrays
  3. Encode using this function
  4. Execute via eth_call RPC (live.udf_api)
  5. Decode response using utils.udf_evm_decode_trace

SUPPORTED TYPES:
  - address: Ethereum addresses
  - uint8, uint16, ..., uint256: Unsigned integers
  - int8, int16, ..., int256: Signed integers
  - bool: Boolean values
  - bytes, bytes1, ..., bytes32: Fixed and dynamic byte arrays
  - string: Dynamic strings
  - Arrays: Any type followed by []
  - Tuples: Nested structures with components
  - Nested combinations: tuple[], tuple[][], etc.

NOTES:
  - Function selector is automatically calculated using Keccak256
  - Compatible with existing utils.udf_evm_text_signature and utils.udf_keccak256
  - Handles gas-optimized function names (e.g., selector 0x00000000)
  - Tuples must be provided as arrays in component order
  - Empty arrays are valid for array-type parameters

ERROR HANDLING:
  - Returns JSON error object on failure
  - Check if result starts with "{" to detect errors
  - Error object includes: error message, traceback, function name, types

RELATED FUNCTIONS:
  - utils.udf_evm_text_signature: Generate function signature
  - utils.udf_keccak256: Calculate function selector
  - utils.udf_evm_decode_trace: Decode call results

{% endmacro %}

{% macro udf_create_eth_call_from_abi_comment() %}
Convenience function that combines contract call encoding and JSON-RPC request creation for eth_call.

PURPOSE:
  Simplifies the workflow of creating eth_call JSON-RPC requests by combining ABI encoding
  and RPC call construction into a single function call. This is the recommended approach for
  most use cases where you want to query contract state via eth_call.

PARAMETERS:
  contract_address (STRING):
    - Ethereum contract address (with or without 0x prefix)
    - Example: '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48'
    
  function_abi (VARIANT):
    - JSON object containing the function ABI definition
    - Must include: "name" (string) and "inputs" (array of input definitions)
    - Each input needs: "name", "type", and optionally "components" for tuples
    - Can be retrieved from tables like crosschain.evm.dim_contract_abis or ethereum.silver.flat_function_abis
    
  input_values (ARRAY):
    - Array of values matching the function inputs in order
    - Values should be provided as native Snowflake types:
      * addresses: strings (with or without 0x prefix)
      * uint/int: numbers
      * bool: booleans
      * bytes/bytes32: hex strings (with or without 0x prefix)
      * arrays: Snowflake arrays
      * tuples: Snowflake arrays in component order
  
  block_parameter (VARIANT, optional):
    - Block identifier for the eth_call request
    - If NULL or omitted: defaults to 'latest'
    - If NUMBER: automatically converted to hex format (e.g., 18500000 -> '0x11a7f80')
    - If STRING: used directly (e.g., 'latest', '0x11a7f80', 'pending')

RETURNS:
  OBJECT: Complete JSON-RPC request object ready for eth_call
    - Format: {"jsonrpc": "2.0", "method": "eth_call", "params": [...], "id": "..."}
    - Can be used directly with RPC execution functions like live.udf_api

EXAMPLES:

  -- Simple balanceOf call with default 'latest' block
  SELECT utils.udf_create_eth_call_from_abi(
    '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
    PARSE_JSON('{
      "name": "balanceOf",
      "inputs": [{"name": "account", "type": "address"}]
    }'),
    ARRAY_CONSTRUCT('0xBcca60bB61934080951369a648Fb03DF4F96263C')
  );

  -- Same call but at a specific block number
  SELECT utils.udf_create_eth_call_from_abi(
    '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
    PARSE_JSON('{
      "name": "balanceOf",
      "inputs": [{"name": "account", "type": "address"}]
    }'),
    ARRAY_CONSTRUCT('0xBcca60bB61934080951369a648Fb03DF4F96263C'),
    18500000
  );

  -- Using ABI from a table
  WITH abi_data AS (
    SELECT 
      abi,
      '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48' as contract_address,
      '0xBcca60bB61934080951369a648Fb03DF4F96263C' as user_address
    FROM ethereum.silver.flat_function_abis
    WHERE contract_address = LOWER('0x43506849D7C04F9138D1A2050bbF3A0c054402dd')
      AND function_name = 'balanceOf'
  )
  SELECT 
    utils.udf_create_eth_call_from_abi(
      contract_address,
      abi,
      ARRAY_CONSTRUCT(user_address)
    ) as rpc_call
  FROM abi_data;

TYPICAL WORKFLOW:
  1. Get function ABI from contract ABI tables (crosschain.evm.dim_contract_abis, etc.)
  2. Prepare input values as Snowflake arrays matching the function signature
  3. Call this function with contract address, ABI, and inputs
  4. Execute the returned RPC call object via live.udf_api or similar
  5. Decode the response using utils.udf_evm_decode_trace or similar decoder

ADVANTAGES OVER MODULAR APPROACH:
  - Single function call instead of two (encode + create)
  - Cleaner, more readable SQL
  - Better for AI systems (fewer steps to explain)
  - Less error-prone (no intermediate variables)
  - More intuitive function name

WHEN TO USE MODULAR FUNCTIONS INSTEAD:
  - When you need to reuse encoded calldata for multiple RPC calls
  - When you need encoded calldata for transaction construction
  - When building complex workflows with intermediate steps

RELATED FUNCTIONS:
  - utils.udf_encode_contract_call: Encode function calls to calldata (used internally)
  - utils.udf_create_eth_call: Create RPC call from encoded calldata (used internally)
  - utils.udf_evm_text_signature: Generate function signature from ABI
  - utils.udf_keccak256: Calculate function selector hash
  - utils.udf_evm_decode_trace: Decode eth_call response results

{% endmacro %}