# 96 B rcf1 table body (fp16 scale + 125 int6 codes) from a trainer joint_state.pt (table_codes, table_scale)
import sys, torch, numpy as np
import hpac_writer135 as HW
st = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
body = HW.encode_table(st["table_codes"].round().numpy().astype(np.int64).reshape(25, 5), float(np.float16(st["table_scale"])))
open(sys.argv[2], "wb").write(body)
print("table body", len(body), "B")
