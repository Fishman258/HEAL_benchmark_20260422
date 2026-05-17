import torch
from torch import nn

from einops import rearrange


class HGTCavAttention(nn.Module):
    def __init__(self, dim, heads, num_types=2,
                 num_relations=4, dim_head=64, dropout=0.1, pose_pe=None):
        super().__init__()
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.num_types = num_types

        self.attend = nn.Softmax(dim=-1)
        self.drop_out = nn.Dropout(dropout)
        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        for t in range(num_types):
            self.k_linears.append(nn.Linear(dim, inner_dim))
            self.q_linears.append(nn.Linear(dim, inner_dim))
            self.v_linears.append(nn.Linear(dim, inner_dim))
            self.a_linears.append(nn.Linear(inner_dim, dim))

        self.relation_att = nn.Parameter(
            torch.Tensor(num_relations, heads, dim_head, dim_head))
        self.relation_msg = nn.Parameter(
            torch.Tensor(num_relations, heads, dim_head, dim_head))

        torch.nn.init.xavier_uniform(self.relation_att)
        torch.nn.init.xavier_uniform(self.relation_msg)

        # Optional pose-aware bias for agent-wise attention.
        pose_cfg = pose_pe or {}
        self.use_pose_pe = bool(pose_cfg.get("enabled", False))
        if self.use_pose_pe:
            hidden_dim = int(pose_cfg.get("hidden_dim", max(32, dim_head * 2)))
            self.pose_trans_scale = float(pose_cfg.get("translation_scale", 50.0))
            in_dim = 9  # (dx,dy,dz) + sin/cos for (roll,yaw,pitch)
            self.pose_mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, heads),
            )
            self.pose_dropout = nn.Dropout(float(pose_cfg.get("dropout", 0.0)))

    def _pose_bias(self, pairwise_t_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pairwise_t_matrix: (B, L, L, 4, 4) where [i,j] maps i->j.
        Returns:
            bias: (B, heads, L, L)
        """
        from opencood.utils.transformation_utils import tfm_to_pose_torch

        B, L = pairwise_t_matrix.shape[:2]
        tfm = pairwise_t_matrix.reshape(-1, 4, 4).to(dtype=torch.float32)
        pose_deg = tfm_to_pose_torch(tfm, dof=6)  # x,y,z,roll,yaw,pitch (degree)

        trans = pose_deg[:, 0:3] / max(self.pose_trans_scale, 1e-6)
        angles = torch.deg2rad(pose_deg[:, 3:6])
        feats = torch.cat([trans, torch.sin(angles), torch.cos(angles)], dim=-1)  # (N, 9)

        bias = self.pose_mlp(feats)  # (N, heads)
        bias = self.pose_dropout(bias)
        bias = bias.view(B, L, L, self.heads).permute(0, 3, 1, 2).contiguous()
        return bias

    def to_qkv(self, x, types):
        # x: (B,H,W,L,C)
        # types: (B,L)
        q_batch = []
        k_batch = []
        v_batch = []

        for b in range(x.shape[0]):
            q_list = []
            k_list = []
            v_list = []

            for i in range(x.shape[-2]):
                # (H,W,1,C)
                q_list.append(
                    self.q_linears[types[b, i]](x[b, :, :, i, :].unsqueeze(2)))
                k_list.append(
                    self.k_linears[types[b, i]](x[b, :, :, i, :].unsqueeze(2)))
                v_list.append(
                    self.v_linears[types[b, i]](x[b, :, :, i, :].unsqueeze(2)))
            # (1,H,W,L,C)
            q_batch.append(torch.cat(q_list, dim=2).unsqueeze(0))
            k_batch.append(torch.cat(k_list, dim=2).unsqueeze(0))
            v_batch.append(torch.cat(v_list, dim=2).unsqueeze(0))
        # (B,H,W,L,C)
        q = torch.cat(q_batch, dim=0)
        k = torch.cat(k_batch, dim=0)
        v = torch.cat(v_batch, dim=0)
        return q, k, v

    def get_relation_type_index(self, type1, type2):
        return type1 * self.num_types + type2

    def get_hetero_edge_weights(self, x, types):
        w_att_batch = []
        w_msg_batch = []

        for b in range(x.shape[0]):
            w_att_list = []
            w_msg_list = []

            for i in range(x.shape[-2]):
                w_att_i_list = []
                w_msg_i_list = []

                for j in range(x.shape[-2]):
                    e_type = self.get_relation_type_index(types[b, i],
                                                          types[b, j])
                    w_att_i_list.append(self.relation_att[e_type].unsqueeze(0))
                    w_msg_i_list.append(self.relation_msg[e_type].unsqueeze(0))
                w_att_list.append(torch.cat(w_att_i_list, dim=0).unsqueeze(0))
                w_msg_list.append(torch.cat(w_msg_i_list, dim=0).unsqueeze(0))

            w_att_batch.append(torch.cat(w_att_list, dim=0).unsqueeze(0))
            w_msg_batch.append(torch.cat(w_msg_list, dim=0).unsqueeze(0))

        # (B,M,L,L,C_head,C_head)
        w_att = torch.cat(w_att_batch, dim=0).permute(0, 3, 1, 2, 4, 5)
        w_msg = torch.cat(w_msg_batch, dim=0).permute(0, 3, 1, 2, 4, 5)
        return w_att, w_msg

    def to_out(self, x, types):
        out_batch = []
        for b in range(x.shape[0]):
            out_list = []
            for i in range(x.shape[-2]):
                out_list.append(
                    self.a_linears[types[b, i]](x[b, :, :, i, :].unsqueeze(2)))
            out_batch.append(torch.cat(out_list, dim=2).unsqueeze(0))
        out = torch.cat(out_batch, dim=0)
        return out

    def forward(self, x, mask, prior_encoding, pairwise_t_matrix=None):
        # x: (B, L, H, W, C) -> (B, H, W, L, C)
        # mask: (B, H, W, L, 1)
        # prior_encoding: (B,L,H,W,3)
        x = x.permute(0, 2, 3, 1, 4)
        # mask: (B, 1, H, W, L, 1)
        mask = mask.unsqueeze(1)
        # (B,L)
        velocities, dts, types = [itm.squeeze(-1) for itm in
                                  prior_encoding[:, :, 0, 0, :].split(
                                      [1, 1, 1], dim=-1)]
        types = types.to(torch.int)
        dts = dts.to(torch.int)
        qkv = self.to_qkv(x, types)
        # (B,M,L,L,C_head,C_head)
        w_att, w_msg = self.get_hetero_edge_weights(x, types)

        # q: (B, M, H, W, L, C)
        q, k, v = map(lambda t: rearrange(t, 'b h w l (m c) -> b m h w l c',
                                          m=self.heads), (qkv))
        # attention, (B, M, H, W, L, L)
        att_map = torch.einsum(
            'b m h w i p, b m i j p q, bm h w j q -> b m h w i j',
            [q, w_att, k]) * self.scale
        if self.use_pose_pe and (pairwise_t_matrix is not None):
            pose_bias = self._pose_bias(pairwise_t_matrix).to(dtype=att_map.dtype, device=att_map.device)
            att_map = att_map + pose_bias[:, :, None, None, :, :]
        # add mask
        att_map = att_map.masked_fill(mask == 0, -float('inf'))
        # softmax
        att_map = self.attend(att_map)

        # out:(B, M, H, W, L, C_head)
        v_msg = torch.einsum('b m i j p c, b m h w j p -> b m h w i j c',
                             w_msg, v)
        out = torch.einsum('b m h w i j, b m h w i j c -> b m h w i c',
                           att_map, v_msg)

        out = rearrange(out, 'b m h w l c -> b h w l (m c)',
                        m=self.heads)
        out = self.to_out(out, types)
        out = self.drop_out(out)
        # (B L H W C)
        out = out.permute(0, 3, 1, 2, 4)
        return out
