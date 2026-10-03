#!/usr/bin/env python3
"""Reproduce semantic_blocks' attributed PR140 lossless derivative.

Learned models, token edits, pose coefficients, and adaptive entropy model are
inherited from adpena's PR140. This encoder groups metadata and applies the
BLK2 packing developed in PR141. All layout information is charged.
"""
from pathlib import Path
import argparse, hashlib, io, json, lzma, struct, sys, zipfile
import brotli
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'cpr1'))
import inflate as renderer
import ddm_mp2_semantic_receiver as schema
from runtime.block_container import decode_model_blocks,transpose
from runtime.segment_layout import pack_segments,unpack_segments

SOURCE_SHA256='cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25'
EXPECTED_SHA256='0e2d95c29e87dca3f9ed14bbc842e57011090ce1e0ac0703c0bb8df3fc70c7e1'

def segment_semantic(blob):
    template=renderer.SemanticTokenRenderer(96).state_dict()
    names=schema._quantized_names(template)
    if blob.startswith(b'SD1M'):
        assert blob[4:6]==bytes((1,len(names)))
        depths,remaining=schema._decode_depth_nibbles(memoryview(blob)[6:],len(names),'SD1M')
        pruned=False
    elif blob.startswith(b'SM3R'):
        assert blob[4]==1 and blob[5] in (5,6) and blob[7]==0
        pruned=True
        mask=int.from_bytes(blob[8:10],'little')
        assert mask==schema._selection_mask(names,schema.ROW_PRUNE_NAMES)
        if blob[5]==6:depths,remaining=schema._decode_depth_nibbles(memoryview(blob)[10:],len(names),'SM3R')
        else:depths,remaining=[4]*len(names),memoryview(blob)[10:]
    else:raise ValueError('Unsupported renderer format')
    allocation=dict(zip(names,depths))
    offset=len(blob)-len(remaining)
    metadata=blob[:offset]
    segments=[]
    for name,value in template.items():
        count=value.numel()
        if value.ndim<2:
            size=count*2
            metadata+=blob[offset:offset+size];offset+=size
            continue
        mask_bytes=0
        scales=schema._scale_count(name,value)
        if pruned and name in schema.ROW_PRUNE_NAMES:
            rows=value.shape[0];mask_bytes=(rows+7)//8
            kept=np.unpackbits(np.frombuffer(blob[offset:offset+mask_bytes],dtype=np.uint8),bitorder='little')[:rows].sum().item()
            assert kept==max(1,round(rows*blob[6]/100))
            count=kept*(count//rows);scales=kept
        size=mask_bytes+scales*2
        metadata+=blob[offset:offset+size];offset+=size
        code_bytes=(count*allocation[name]+7)//8
        codes=blob[offset:offset+code_bytes];offset+=code_bytes
        segments.extend((metadata,codes));metadata=b''
    if metadata:segments.append(metadata)
    assert offset==len(blob) and b''.join(segments)==blob
    packed=pack_segments(segments)
    assert unpack_segments(packed)==blob
    metadata_end=6+2*len(segments)+sum(map(len,segments[::2]))
    return packed,metadata_end,[len(s) for s in segments]

def build_payload(models: bytes, tail: bytes, recipe: dict) -> bytes:
    blocks = []
    offset = 0
    codecs = {'raw': 0, 'lzma': 1, 'brotli': 2}
    for span in recipe['blocks']:
        if span['start'] != offset or not offset < span['end'] <= len(models):
            raise ValueError('Noncontiguous or out-of-range recipe')
        data = models[offset:span['end']]
        transform = b''
        codec = codecs[span['codec']]
        kind = span.get('kind', 'identity')
        if kind != 'identity':
            name, stride = kind.split(':')
            if name != 'transpose':
                raise ValueError('Unsupported transform')
            stride = int(stride)
            data = transpose(data, stride)
            transform = bytes((1, stride))
        if codec == 1:
            data = lzma.compress(data, format=lzma.FORMAT_RAW, filters=[span['params']])
        elif codec == 2:
            import brotli
            data = brotli.compress(data, **span['params'])
        blocks.append(bytes((codec | (128 if transform else 0),)) + len(data).to_bytes(3, 'little') + transform + data)
        offset = span['end']
    if offset != len(models):
        raise ValueError('Recipe does not cover all model data')
    payload = b'BLK2' + struct.pack('<IB', len(models), len(blocks)) + b''.join(blocks) + tail
    if decode_model_blocks(payload) != (models, tail):
        raise ValueError('Lossless round trip failed')
    return payload

def zip_payload(payload: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as z:
        info = zipfile.ZipInfo('p', date_time=(1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.external_attr = 0o644 << 16
        z.writestr(info, payload)
    return output.getvalue()

def build(source):
    if hashlib.sha256(source).hexdigest()!=SOURCE_SHA256:
        raise ValueError('Expected the pinned 180,002-byte PR140 archive; see REPRODUCING.md')
    with zipfile.ZipFile(io.BytesIO(source)) as z:
        if z.namelist()!=['p']:raise ValueError('Expected only ZIP member p')
        payload=z.read('p')
    header=struct.Struct('<4sBBBBHHH')
    magic,version,codec,table,flags,*sizes=header.unpack_from(payload)
    if (magic,version,codec,table,flags)!=(b'RX1M',1,2,0,26):
        raise ValueError('Unexpected source representation')
    offset=header.size;streams=[]
    for size in sizes:
        streams.append(payload[offset:offset+size]);offset+=size
    tail=payload[offset:]
    semantic=transpose(brotli.decompress(streams[1]),2,inverse=True)
    grouped,_,_=segment_semantic(semantic)
    bodies=[streams[0],grouped,brotli.decompress(streams[2])]
    models=header.pack(magic,version,codec,table,flags&~2,*map(len,bodies))+b''.join(bodies)
    recipe=json.loads((HERE/'recipe.json').read_text())
    result=zip_payload(build_payload(models,tail,recipe))
    if hashlib.sha256(result).hexdigest()!=EXPECTED_SHA256:
        raise ValueError('Rebuilt archive differs from the validated candidate')
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=HERE/'rebuilt.zip')
    args=ap.parse_args()
    if args.output.resolve()==args.source.resolve():raise ValueError('Source and output must differ')
    source=args.source.read_bytes()
    result=build(source)
    if result!=build(source):raise ValueError('Repeated builds differ')
    if args.output.exists():raise FileExistsError('Choose a fresh output path')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes(result)
    print(json.dumps(dict(archive_bytes=len(result),sha256=hashlib.sha256(result).hexdigest(),source_bytes=len(source),saved_bytes=len(source)-len(result)),indent=2))

if __name__=='__main__':main()
