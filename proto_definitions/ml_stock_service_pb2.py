# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: proto_definitions/ml_stock_service.proto
# Protobuf Python Version: 5.29.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    2,
    '',
    'proto_definitions/ml_stock_service.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n(proto_definitions/ml_stock_service.proto\"G\n\x0eMLStockRequest\x12\x0f\n\x07symbols\x18\x01 \x03(\t\x12\x12\n\nstart_date\x18\x02 \x01(\t\x12\x10\n\x08\x65nd_date\x18\x03 \x01(\t\"5\n\x0fMLStockResponse\x12\"\n\x0bsymbol_data\x18\x01 \x03(\x0b\x32\r.MLSymbolData\"Q\n\x0cMLSymbolData\x12\x0e\n\x06symbol\x18\x01 \x01(\t\x12 \n\ndaily_data\x18\x02 \x03(\x0b\x32\x0c.MLDailyData\x12\x0f\n\x07signals\x18\x03 \x03(\t\"c\n\x0bMLDailyData\x12\x0c\n\x04\x64\x61te\x18\x01 \x01(\t\x12\x0c\n\x04open\x18\x02 \x01(\x02\x12\x0c\n\x04high\x18\x03 \x01(\x02\x12\x0b\n\x03low\x18\x04 \x01(\x02\x12\r\n\x05\x63lose\x18\x05 \x01(\x02\x12\x0e\n\x06volume\x18\x06 \x01(\x03\x32\x45\n\x0eMLStockService\x12\x33\n\x0eGetMLStockData\x12\x0f.MLStockRequest\x1a\x10.MLStockResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'proto_definitions.ml_stock_service_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_MLSTOCKREQUEST']._serialized_start=44
  _globals['_MLSTOCKREQUEST']._serialized_end=115
  _globals['_MLSTOCKRESPONSE']._serialized_start=117
  _globals['_MLSTOCKRESPONSE']._serialized_end=170
  _globals['_MLSYMBOLDATA']._serialized_start=172
  _globals['_MLSYMBOLDATA']._serialized_end=253
  _globals['_MLDAILYDATA']._serialized_start=255
  _globals['_MLDAILYDATA']._serialized_end=354
  _globals['_MLSTOCKSERVICE']._serialized_start=356
  _globals['_MLSTOCKSERVICE']._serialized_end=425
# @@protoc_insertion_point(module_scope)
