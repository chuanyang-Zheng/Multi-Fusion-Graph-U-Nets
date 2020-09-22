# import torch
# import torch.nn as nn
# import numpy as np
#
#
# class GraphUnet(nn.Module):
#
#     def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
#         super(GraphUnet, self).__init__()
#         self.ks = ks
#         self.bottom_gcn = GCN(dim, dim, act, drop_p)
#         self.down_gcns = nn.ModuleList()
#         self.up_gcns = nn.ModuleList()
#         self.pools = nn.ModuleList()
#         self.unpools = nn.ModuleList()
#         self.l_n = len(ks)
#         for i in range(self.l_n):
#             self.down_gcns.append(GCN(dim, dim, act, drop_p))
#             self.up_gcns.append(GCN(dim, dim, act, drop_p))
#             self.pools.append(Pool(ks[i], dim, drop_p))
#             self.unpools.append(Unpool(dim, dim, drop_p))
#         self.fuselayer_1=FuseLayer(dim,act,drop_p,self.l_n+1)
#         # self.fuselayer_2 = FuseLayer(dim, act, drop_p, self.l_n + 1)
#         # self.fuselayer_3 = FuseLayer(dim, act, drop_p, self.l_n + 1)
#         # self.fuselayer_4 = FuseLayer(dim, act, drop_p, self.l_n + 1)
#         # self.GCNList_1=GCNList(dim,act,drop_p,self.l_n+1)
#         # self.GCNList_2 = GCNList(dim, act, drop_p, self.l_n + 1)
#
#
#
#     def forward(self, g, h):
#         adj_ms = []
#         indices_list = []
#         down_outs = []
#         hs = []
#         org_h = h
#         org_g=g
#         for i in range(self.l_n):
#             h = self.down_gcns[i](g, h)
#             adj_ms.append(g)
#             down_outs.append(h)
#             g, h, idx = self.pools[i](g, h)
#
#             indices_list.append(idx)
#         adj_ms.append(g)
#         down_outs.append(h)
#
#         # down_outs=self.GCNList_1(adj_ms,down_outs)
#         down_outs_fuse=self.fuselayer_1(adj_ms,down_outs,indices_list)
#         # down_outs = self.fuselayer_2(adj_ms, down_outs, indices_list)
#         # down_outs = self.fuselayer_3(adj_ms, down_outs, indices_list)
#         # down_outs = self.fuselayer_4(adj_ms, down_outs, indices_list)
#
#         # down_outs = self.GCNList_2(adj_ms, down_outs, indices_list)
#
#         # h=down_outs[self.l_n]
#         h = down_outs[self.l_n]+down_outs_fuse[self.l_n]+self.bottom_gcn(adj_ms[self.l_n], down_outs_fuse[self.l_n])
#         for i in range(self.l_n):
#             up_idx = self.l_n - i - 1
#             g, idx = adj_ms[up_idx], indices_list[up_idx]
#             g, h = self.unpools[i](g, h, down_outs_fuse[up_idx], idx)
#             h = self.up_gcns[i](g, h)
#             h = h.add(down_outs[up_idx])
#             h=h.add(down_outs_fuse[up_idx])
#             hs.append(h)
#         h = h.add(org_h)
#         hs.append(h)
#         return hs
#
#
#
#
# class GCN(nn.Module):
#
#     def __init__(self, in_dim, out_dim, act, p):
#         super(GCN, self).__init__()
#         self.proj = nn.Linear(in_dim, out_dim)
#         self.act = act
#         self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
#
#     def forward(self, g, h):
#         h = self.drop(h)
#         h = torch.matmul(g, h)
#         h = self.proj(h)
#         h = self.act(h)
#         return h
#
#
# class Pool(nn.Module):
#
#     def __init__(self, k, in_dim, p):
#         super(Pool, self).__init__()
#         self.k = k
#         self.sigmoid = nn.Sigmoid()
#         self.proj = nn.Linear(in_dim, 1)
#         self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
#
#     def forward(self, g, h):
#         Z = self.drop(h)
#         weights = self.proj(Z).squeeze()
#         scores = self.sigmoid(weights)
#         return top_k_graph(scores, g, h, self.k)
#
#
# class Unpool(nn.Module):
#
#     def __init__(self, *args):
#         super(Unpool, self).__init__()
#
#     def forward(self, g, h, pre_h, idx):
#         new_h = h.new_zeros([g.shape[0], h.shape[1]])
#         new_h[idx] = h
#         return g, new_h
#
# class Up_One(nn.Module):
#     def __init__(self,dim, act, drop_p):
#         super(Up_One, self).__init__()
#         # self.gcn=GCN(dim,dim,act,drop_p)
#     def forward(self,target_g, original_level_h,  original_level_idx):
#         new_h = original_level_h.new_zeros([target_g.shape[0], original_level_h.shape[1]])
#         new_h[original_level_idx] = original_level_h
#         # new_h=self.gcn(target_g,new_h)
#         return new_h
#
#
# class Up(nn.Module):
#     def __init__(self,dim, act, drop_p,begin_level,final_level ):
#         super(Up, self).__init__()
#         # self.gcn=GCN(in_dim, out_dim, act, p)
#         self.begin_level=begin_level
#         self.final_level=final_level
#         proess_list=nn.ModuleList()
#         for i in range(begin_level-final_level):
#             proess_list.append(Up_One(dim, act, drop_p))
#         self.process_list=proess_list
#
#         self.gcn = GCN(dim, dim, act, drop_p)
#
#     def forward(self, adj_ms,down_outs, idx_list):
#         # process_h = self.gcn(original_level_g,original_level_h)
#         h_value=down_outs[self.begin_level]
#         for i in range(self.begin_level-self.final_level):
#             h_value=self.process_list[i](adj_ms[self.begin_level-i-1],h_value,idx_list[self.begin_level-i-1])
#         h_value=self.gcn(adj_ms[self.final_level],h_value)
#
#
#         return h_value
#
#
# class Down_One(nn.Module):
#     def __init__(self, in_dim, out_dim, act, p):
#         super(Down_One, self).__init__()
#         self.sigmoid = nn.Sigmoid()
#         self.proj = nn.Linear(in_dim, 1)
#         self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
#         # self.gcn=GCN(in_dim, out_dim, act, p)
#     def forward(self, this_level_g, this_level_h,idx):
#         # h=self.gcn(this_level_g,this_level_h)
#         # Z = self.drop(h)
#         # weights = self.proj(Z).squeeze()
#         # scores = self.sigmoid(weights)
#         # values=scores[idx]
#         # new_h = h[idx, :]
#         # values = torch.unsqueeze(values, -1)
#         # new_h = torch.mul(new_h, values)
#
#         new_h=this_level_h[idx,:]
#         return new_h
#
# class Down_Multi(nn.Module):
#     def __init__(self, in_dim, out_dim, act, p,begin_level,final_level):
#         super(Down_Multi, self).__init__()
#         process=nn.ModuleList()
#         for i in range(final_level-begin_level):
#             process.append(Down_One(in_dim, out_dim, act, p))
#         self.process=process
#         self.begin_level=begin_level
#         self.final_level=final_level
#         self.gcn = GCN(in_dim, out_dim, act, p)
#     def forward(self, adj_ms,down_outs, idx_list):
#         h_value = down_outs[self.begin_level]
#         for i in range(self.final_level-self.begin_level):
#
#             h_value = self.process[i](adj_ms[self.begin_level+i ], h_value, idx_list[self.begin_level+i])
#             # if (self.final_level == 2 and self.begin_level == 0):
#             #     print("H Value Size {}".format(h_value.size()))
#         h_value=self.gcn(adj_ms[self.final_level],h_value)
#         return h_value
#
#
# class FuseLayer(nn.Module):
#     def __init__(self,dim, act, drop_p,list_length):
#         super(FuseLayer, self).__init__()
#         self.list_length=list_length
#         processList = nn.ModuleList()
#         for i in range(self.list_length):
#             temp = nn.ModuleList()
#             for j in range(self.list_length):
#                 if i == j:
#                     temp.append(GCN(dim, dim, act, drop_p))
#                 elif i < j:
#                     temp.append(Up(dim,act,drop_p,j, i))
#                 else:
#                     temp.append(Down_Multi(dim, dim, act, drop_p, j, i))
#             processList.append(temp)
#         self.processList = processList
#     def forward(self,adj_ms,down_outs, idx_list):
#         result=[]
#         for i in range(self.list_length):
#             temp=0
#             for j in range(self.list_length):
#                 if i==j:
#                     calculate=self.processList[i][j](adj_ms[i],down_outs[i])
#                     # print("Directly From {} to {} result size {}".format(j,i,calculate.size()))
#                     temp+=calculate
#                 else:
#                     calculate = self.processList[i][j](adj_ms, down_outs,idx_list)
#                     # print("From {} to {} result size {}".format(j,i,calculate.size()))
#                     temp += calculate
#             result.append(temp)
#         return result
# class GCNList(nn.Module):
#     def __init__(self,dim, act, drop_p,list_length):
#         super(GCNList, self).__init__()
#         process=nn.ModuleList()
#         self.list_length=list_length
#         for i in range(list_length):
#             process.append(GCN(dim, dim, act, drop_p))
#         self.process=process
#     def forward(self,adj_ms,down_outs):
#         temp_result=[]
#         for i in range(self.list_length):
#             temp_result.append(self.process[i](adj_ms[i],down_outs[i]))
#         return temp_result
#
# def top_k_graph(scores, g, h, k):
#     num_nodes = g.shape[0]
#     values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
#     new_h = h[idx, :]
#     values = torch.unsqueeze(values, -1)
#     new_h = torch.mul(new_h, values)
#     un_g = g.bool().float()
#     un_g = torch.matmul(un_g, un_g).bool().float()
#     un_g = un_g[idx, :]
#     un_g = un_g[:, idx]
#     g = norm_g(un_g)
#     return g, new_h, idx
#
#
# def norm_g(g):
#     degrees = torch.sum(g, 1)
#     g = g / degrees
#     return g
#
#
# class Initializer(object):
#
#     @classmethod
#     def _glorot_uniform(cls, w):
#         if len(w.size()) == 2:
#             fan_in, fan_out = w.size()
#         elif len(w.size()) == 3:
#             fan_in = w.size()[1] * w.size()[2]
#             fan_out = w.size()[0] * w.size()[2]
#         else:
#             fan_in = np.prod(w.size())
#             fan_out = np.prod(w.size())
#         limit = np.sqrt(6.0 / (fan_in + fan_out))
#         w.uniform_(-limit, limit)
#
#     @classmethod
#     def _param_init(cls, m):
#         if isinstance(m, nn.parameter.Parameter):
#             cls._glorot_uniform(m.data)
#         elif isinstance(m, nn.Linear):
#             m.bias.data.zero_()
#             cls._glorot_uniform(m.weight.data)
#
#     @classmethod
#     def weights_init(cls, m):
#         for p in m.modules():
#             if isinstance(p, nn.ParameterList):
#                 for pp in p:
#                     cls._param_init(pp)
#             else:
#                 cls._param_init(p)
#
#         for name, p in m.named_parameters():
#             if '.' not in name:
#                 cls._param_init(p)





# import torch
# import torch.nn as nn
# import numpy as np
#
#
# class GraphUnet(nn.Module):
#
#     def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
#         super(GraphUnet, self).__init__()
#         self.ks = ks
#         self.bottom_gcn = GCN(dim, dim, act, drop_p)
#         self.down_gcns = nn.ModuleList()
#         self.up_gcns = nn.ModuleList()
#         self.pools = nn.ModuleList()
#         self.unpools = nn.ModuleList()
#         self.l_n = len(ks)
#         for i in range(self.l_n):
#             self.down_gcns.append(GCN(dim, dim, act, drop_p))
#             self.up_gcns.append(GCN(dim, dim, act, drop_p))
#             self.pools.append(Pool(ks[i], dim, drop_p))
#             self.unpools.append(Unpool(dim, dim, drop_p))
#         self.fuselayer_1=FuseLayer(dim,act,drop_p,self.l_n+1)
#         # self.fuselayer_2 = FuseLayer(dim, act, drop_p, self.l_n + 1)
#         # self.fuselayer_3 = FuseLayer(dim, act, drop_p, self.l_n + 1)
#         # self.fuselayer_4 = FuseLayer(dim, act, drop_p, self.l_n + 1)
#         # self.GCNList_1=GCNList(dim,act,drop_p,self.l_n+1)
#         # self.GCNList_2 = GCNList(dim, act, drop_p, self.l_n + 1)
#
#
#
#     def forward(self, g, h):
#         adj_ms = []
#         indices_list = []
#         down_outs = []
#         hs = []
#         org_h = h
#         org_g=g
#         for i in range(self.l_n):
#             h = self.down_gcns[i](g, h)
#             adj_ms.append(g)
#             down_outs.append(h)
#             g, h, idx = self.pools[i](g, h)
#
#             indices_list.append(idx)
#         adj_ms.append(g)
#         down_outs.append(h)
#
#         # down_outs=self.GCNList_1(adj_ms,down_outs)
#         down_outs_fuse=self.fuselayer_1(adj_ms,down_outs,indices_list)
#         # down_outs = self.fuselayer_2(adj_ms, down_outs, indices_list)
#         # down_outs = self.fuselayer_3(adj_ms, down_outs, indices_list)
#         # down_outs = self.fuselayer_4(adj_ms, down_outs, indices_list)
#
#         # down_outs = self.GCNList_2(adj_ms, down_outs, indices_list)
#
#         # h=down_outs[self.l_n]
#         h = self.bottom_gcn(adj_ms[self.l_n], down_outs_fuse[self.l_n])
#         for i in range(self.l_n):
#             up_idx = self.l_n - i - 1
#             g, idx = adj_ms[up_idx], indices_list[up_idx]
#             g, h = self.unpools[i](g, down_outs[self.l_n-i]+down_outs_fuse[self.l_n-i]+h, down_outs_fuse[up_idx], idx)
#             h = self.up_gcns[i](g, h)
#             h = h.add(down_outs[up_idx])
#             hs.append(h)
#         h = h.add(org_h)
#         hs.append(h)
#         return hs
#
#
#
#
# class GCN(nn.Module):
#
#     def __init__(self, in_dim, out_dim, act, p):
#         super(GCN, self).__init__()
#         self.proj = nn.Linear(in_dim, out_dim)
#         self.act = act
#         self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
#
#     def forward(self, g, h):
#         h = self.drop(h)
#         h = torch.matmul(g, h)
#         h = self.proj(h)
#         h = self.act(h)
#         return h
#
#
# class Pool(nn.Module):
#
#     def __init__(self, k, in_dim, p):
#         super(Pool, self).__init__()
#         self.k = k
#         self.sigmoid = nn.Sigmoid()
#         self.proj = nn.Linear(in_dim, 1)
#         self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
#
#     def forward(self, g, h):
#         Z = self.drop(h)
#         weights = self.proj(Z).squeeze()
#         scores = self.sigmoid(weights)
#         return top_k_graph(scores, g, h, self.k)
#
#
# class Unpool(nn.Module):
#
#     def __init__(self, *args):
#         super(Unpool, self).__init__()
#
#     def forward(self, g, h, pre_h, idx):
#         new_h = h.new_zeros([g.shape[0], h.shape[1]])
#         new_h[idx] = h
#         return g, new_h
#
# class Up_One(nn.Module):
#     def __init__(self,dim, act, drop_p):
#         super(Up_One, self).__init__()
#         self.gcn=GCN(dim,dim,act,drop_p)
#     def forward(self,target_g, original_level_h,  original_level_idx):
#         new_h = original_level_h.new_zeros([target_g.shape[0], original_level_h.shape[1]])
#         new_h[original_level_idx] = original_level_h
#         new_h=self.gcn(target_g,new_h)
#         return new_h
#
#
# class Up(nn.Module):
#     def __init__(self,dim, act, drop_p,begin_level,final_level ):
#         super(Up, self).__init__()
#         # self.gcn=GCN(in_dim, out_dim, act, p)
#         self.begin_level=begin_level
#         self.final_level=final_level
#         proess_list=nn.ModuleList()
#         for i in range(begin_level-final_level):
#             proess_list.append(Up_One(dim, act, drop_p))
#         self.process_list=proess_list
#
#     def forward(self, adj_ms,down_outs, idx_list):
#         # process_h = self.gcn(original_level_g,original_level_h)
#         h_value=down_outs[self.begin_level]
#         for i in range(self.begin_level-self.final_level):
#             h_value=self.process_list[i](adj_ms[self.begin_level-i-1],h_value,idx_list[self.begin_level-i-1])
#
#
#         return h_value
#
#
# class Down_One(nn.Module):
#     def __init__(self, in_dim, out_dim, act, p):
#         super(Down_One, self).__init__()
#         self.sigmoid = nn.Sigmoid()
#         self.proj = nn.Linear(in_dim, 1)
#         self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
#         self.gcn=GCN(in_dim, out_dim, act, p)
#     def forward(self, this_level_g, this_level_h,idx):
#         h=self.gcn(this_level_g,this_level_h)
#         Z = self.drop(h)
#         weights = self.proj(Z).squeeze()
#         scores = self.sigmoid(weights)
#         values=scores[idx]
#         new_h = h[idx, :]
#         values = torch.unsqueeze(values, -1)
#         new_h = torch.mul(new_h, values)
#         return new_h
#
# class Down_Multi(nn.Module):
#     def __init__(self, in_dim, out_dim, act, p,begin_level,final_level):
#         super(Down_Multi, self).__init__()
#         process=nn.ModuleList()
#         for i in range(final_level-begin_level):
#             process.append(Down_One(in_dim, out_dim, act, p))
#         self.process=process
#         self.begin_level=begin_level
#         self.final_level=final_level
#     def forward(self, adj_ms,down_outs, idx_list):
#         h_value = down_outs[self.begin_level]
#         for i in range(self.final_level-self.begin_level):
#
#             h_value = self.process[i](adj_ms[self.begin_level+i ], h_value, idx_list[self.begin_level+i])
#             # if (self.final_level == 2 and self.begin_level == 0):
#             #     print("H Value Size {}".format(h_value.size()))
#         return h_value
#
#
# class FuseLayer(nn.Module):
#     def __init__(self,dim, act, drop_p,list_length):
#         super(FuseLayer, self).__init__()
#         self.list_length=list_length
#         processList = nn.ModuleList()
#         for i in range(self.list_length):
#             temp = nn.ModuleList()
#             for j in range(self.list_length):
#                 if i == j:
#                     temp.append(GCN(dim, dim, act, drop_p))
#                 elif i < j:
#                     temp.append(Up(dim,act,drop_p,j, i))
#                 else:
#                     temp.append(Down_Multi(dim, dim, act, drop_p, j, i))
#             processList.append(temp)
#         self.processList = processList
#     def forward(self,adj_ms,down_outs, idx_list):
#         result=[]
#         for i in range(self.list_length):
#             temp=0
#             for j in range(self.list_length):
#                 if i==j:
#                     calculate=self.processList[i][j](adj_ms[i],down_outs[i])
#                     # print("Directly From {} to {} result size {}".format(j,i,calculate.size()))
#                     temp+=calculate
#                 else:
#                     calculate = self.processList[i][j](adj_ms, down_outs,idx_list)
#                     # print("From {} to {} result size {}".format(j,i,calculate.size()))
#                     temp += calculate
#             result.append(temp)
#         return result
# class GCNList(nn.Module):
#     def __init__(self,dim, act, drop_p,list_length):
#         super(GCNList, self).__init__()
#         process=nn.ModuleList()
#         self.list_length=list_length
#         for i in range(list_length):
#             process.append(GCN(dim, dim, act, drop_p))
#         self.process=process
#     def forward(self,adj_ms,down_outs):
#         temp_result=[]
#         for i in range(self.list_length):
#             temp_result.append(self.process[i](adj_ms[i],down_outs[i]))
#         return temp_result
#
# def top_k_graph(scores, g, h, k):
#     num_nodes = g.shape[0]
#     values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
#     new_h = h[idx, :]
#     values = torch.unsqueeze(values, -1)
#     new_h = torch.mul(new_h, values)
#     un_g = g.bool().float()
#     un_g = torch.matmul(un_g, un_g).bool().float()
#     un_g = un_g[idx, :]
#     un_g = un_g[:, idx]
#     g = norm_g(un_g)
#     return g, new_h, idx
#
#
# def norm_g(g):
#     degrees = torch.sum(g, 1)
#     g = g / degrees
#     return g
#
#
# class Initializer(object):
#
#     @classmethod
#     def _glorot_uniform(cls, w):
#         if len(w.size()) == 2:
#             fan_in, fan_out = w.size()
#         elif len(w.size()) == 3:
#             fan_in = w.size()[1] * w.size()[2]
#             fan_out = w.size()[0] * w.size()[2]
#         else:
#             fan_in = np.prod(w.size())
#             fan_out = np.prod(w.size())
#         limit = np.sqrt(6.0 / (fan_in + fan_out))
#         w.uniform_(-limit, limit)
#
#     @classmethod
#     def _param_init(cls, m):
#         if isinstance(m, nn.parameter.Parameter):
#             cls._glorot_uniform(m.data)
#         elif isinstance(m, nn.Linear):
#             m.bias.data.zero_()
#             cls._glorot_uniform(m.weight.data)
#
#     @classmethod
#     def weights_init(cls, m):
#         for p in m.modules():
#             if isinstance(p, nn.ParameterList):
#                 for pp in p:
#                     cls._param_init(pp)
#             else:
#                 cls._param_init(p)
#
#         for name, p in m.named_parameters():
#             if '.' not in name:
#                 cls._param_init(p)

# #Best
# import torch
# import torch.nn as nn
# import numpy as np
#
#
# class GraphUnet(nn.Module):
#
#     def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
#         super(GraphUnet, self).__init__()
#         self.ks = ks
#         self.bottom_gcn = GCN(dim, dim, act, drop_p)
#         self.down_gcns = nn.ModuleList()
#         self.up_gcns = nn.ModuleList()
#         self.pools = nn.ModuleList()
#         self.unpools = nn.ModuleList()
#         self.l_n = len(ks)
#         for i in range(self.l_n):
#             self.down_gcns.append(GCN(dim, dim, act, drop_p))
#             self.up_gcns.append(GCN(dim, dim, act, drop_p))
#             self.pools.append(Pool(ks[i], dim, drop_p))
#             self.unpools.append(Unpool(dim, dim, drop_p))
#         self.fuselayer_1=FuseLayer(dim,act,drop_p,self.l_n+1)
#         # self.fuselayer_2 = FuseLayer(dim, act, drop_p, self.l_n + 1)
#         # self.fuselayer_3 = FuseLayer(dim, act, drop_p, self.l_n + 1)
#         # self.fuselayer_4 = FuseLayer(dim, act, drop_p, self.l_n + 1)
#         # self.GCNList_1=GCNList(dim,act,drop_p,self.l_n+1)
#         # self.GCNList_2 = GCNList(dim, act, drop_p, self.l_n + 1)
#
#
#
#     def forward(self, g, h):
#         adj_ms = []
#         indices_list = []
#         down_outs = []
#         hs = []
#         org_h = h
#         org_g=g
#         for i in range(self.l_n):
#             h = self.down_gcns[i](g, h)
#             adj_ms.append(g)
#             down_outs.append(h)
#             g, h, idx = self.pools[i](g, h)
#
#             indices_list.append(idx)
#         adj_ms.append(g)
#         down_outs.append(h)
#
#         # down_outs=self.GCNList_1(adj_ms,down_outs)
#         down_outs_fuse=self.fuselayer_1(adj_ms,down_outs,indices_list)
#         # down_outs = self.fuselayer_2(adj_ms, down_outs, indices_list)
#         # down_outs = self.fuselayer_3(adj_ms, down_outs, indices_list)
#         # down_outs = self.fuselayer_4(adj_ms, down_outs, indices_list)
#
#         # down_outs = self.GCNList_2(adj_ms, down_outs, indices_list)
#
#         # h=down_outs[self.l_n]
#         # h = down_outs[self.l_n]+down_outs_fuse[self.l_n]+self.bottom_gcn(adj_ms[self.l_n], down_outs_fuse[self.l_n])
#         # for i in range(self.l_n):
#         #     up_idx = self.l_n - i - 1
#         #     g, idx = adj_ms[up_idx], indices_list[up_idx]
#         #     g, h = self.unpools[i](g, h, down_outs_fuse[up_idx], idx)
#         #     h = self.up_gcns[i](g, h)
#         #     h = h.add(down_outs[up_idx])
#         #     h=h.add(down_outs_fuse[up_idx])
#         #     hs.append(h)
#         # h = h.add(org_h)
#         # hs.append(h)
#
#         down_outs_fuse[0]=down_outs_fuse[0]+org_h
#         return down_outs_fuse
#
#
#
#
# class GCN(nn.Module):
#
#     def __init__(self, in_dim, out_dim, act, p):
#         super(GCN, self).__init__()
#         self.proj = nn.Linear(in_dim, out_dim)
#         self.act = act
#         self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
#
#     def forward(self, g, h):
#         h = self.drop(h)
#         h = torch.matmul(g, h)
#         h = self.proj(h)
#         h = self.act(h)
#         return h
#
#
# class Pool(nn.Module):
#
#     def __init__(self, k, in_dim, p):
#         super(Pool, self).__init__()
#         self.k = k
#         self.sigmoid = nn.Sigmoid()
#         self.proj = nn.Linear(in_dim, 1)
#         self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
#
#     def forward(self, g, h):
#         Z = self.drop(h)
#         weights = self.proj(Z).squeeze()
#         scores = self.sigmoid(weights)
#         return top_k_graph(scores, g, h, self.k)
#
#
# class Unpool(nn.Module):
#
#     def __init__(self, *args):
#         super(Unpool, self).__init__()
#
#     def forward(self, g, h, pre_h, idx):
#         new_h = h.new_zeros([g.shape[0], h.shape[1]])
#         new_h[idx] = h
#         return g, new_h
#
# class Up_One(nn.Module):
#     def __init__(self,dim, act, drop_p):
#         super(Up_One, self).__init__()
#         # self.gcn=GCN(dim,dim,act,drop_p)
#     def forward(self,target_g, original_level_h,  original_level_idx):
#         new_h = original_level_h.new_zeros([target_g.shape[0], original_level_h.shape[1]])
#         new_h[original_level_idx] = original_level_h
#         # new_h=self.gcn(target_g,new_h)
#         return new_h
#
#
# class Up(nn.Module):
#     def __init__(self,dim, act, drop_p,begin_level,final_level ):
#         super(Up, self).__init__()
#         # self.gcn=GCN(in_dim, out_dim, act, p)
#         self.begin_level=begin_level
#         self.final_level=final_level
#         proess_list=nn.ModuleList()
#         for i in range(begin_level-final_level):
#             proess_list.append(Up_One(dim, act, drop_p))
#         self.process_list=proess_list
#
#         self.gcn = GCN(dim, dim, act, drop_p)
#
#     def forward(self, adj_ms,down_outs, idx_list):
#         # process_h = self.gcn(original_level_g,original_level_h)
#         h_value=down_outs[self.begin_level]
#         for i in range(self.begin_level-self.final_level):
#             h_value=self.process_list[i](adj_ms[self.begin_level-i-1],h_value,idx_list[self.begin_level-i-1])
#         h_value=self.gcn(adj_ms[self.final_level],h_value)
#
#
#         return h_value
#
#
# class Down_One(nn.Module):
#     def __init__(self, in_dim, out_dim, act, p):
#         super(Down_One, self).__init__()
#         self.sigmoid = nn.Sigmoid()
#         self.proj = nn.Linear(in_dim, 1)
#         self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
#         # self.gcn=GCN(in_dim, out_dim, act, p)
#     def forward(self, this_level_g, this_level_h,idx):
#         # h=self.gcn(this_level_g,this_level_h)
#         # Z = self.drop(h)
#         # weights = self.proj(Z).squeeze()
#         # scores = self.sigmoid(weights)
#         # values=scores[idx]
#         # new_h = h[idx, :]
#         # values = torch.unsqueeze(values, -1)
#         # new_h = torch.mul(new_h, values)
#
#         new_h=this_level_h[idx,:]
#         return new_h
#
# class Down_Multi(nn.Module):
#     def __init__(self, in_dim, out_dim, act, p,begin_level,final_level):
#         super(Down_Multi, self).__init__()
#         process=nn.ModuleList()
#         for i in range(final_level-begin_level):
#             process.append(Down_One(in_dim, out_dim, act, p))
#         self.process=process
#         self.begin_level=begin_level
#         self.final_level=final_level
#         self.gcn = GCN(in_dim, out_dim, act, p)
#     def forward(self, adj_ms,down_outs, idx_list):
#         h_value = down_outs[self.begin_level]
#         for i in range(self.final_level-self.begin_level):
#
#             h_value = self.process[i](adj_ms[self.begin_level+i ], h_value, idx_list[self.begin_level+i])
#             # if (self.final_level == 2 and self.begin_level == 0):
#             #     print("H Value Size {}".format(h_value.size()))
#         h_value=self.gcn(adj_ms[self.final_level],h_value)
#         return h_value
#
#
# class FuseLayer(nn.Module):
#     def __init__(self,dim, act, drop_p,list_length):
#         super(FuseLayer, self).__init__()
#         self.list_length=list_length
#         processList = nn.ModuleList()
#         for i in range(self.list_length):
#             temp = nn.ModuleList()
#             for j in range(self.list_length):
#                 if i == j:
#                     temp.append(GCN(dim, dim, act, drop_p))
#                 elif i < j:
#                     temp.append(Up(dim,act,drop_p,j, i))
#                 else:
#                     temp.append(Down_Multi(dim, dim, act, drop_p, j, i))
#             processList.append(temp)
#         self.processList = processList
#     def forward(self,adj_ms,down_outs, idx_list):
#         result=[]
#         for i in range(self.list_length):
#             temp=0
#             for j in range(self.list_length):
#                 if i==j:
#                     calculate=self.processList[i][j](adj_ms[i],down_outs[i])
#                     # print("Directly From {} to {} result size {}".format(j,i,calculate.size()))
#                     temp+=calculate
#                 else:
#                     calculate = self.processList[i][j](adj_ms, down_outs,idx_list)
#                     # print("From {} to {} result size {}".format(j,i,calculate.size()))
#                     temp += calculate
#             result.append(temp)
#         return result
# class GCNList(nn.Module):
#     def __init__(self,dim, act, drop_p,list_length):
#         super(GCNList, self).__init__()
#         process=nn.ModuleList()
#         self.list_length=list_length
#         for i in range(list_length):
#             process.append(GCN(dim, dim, act, drop_p))
#         self.process=process
#     def forward(self,adj_ms,down_outs):
#         temp_result=[]
#         for i in range(self.list_length):
#             temp_result.append(self.process[i](adj_ms[i],down_outs[i]))
#         return temp_result
#
# def top_k_graph(scores, g, h, k):
#     num_nodes = g.shape[0]
#     values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
#     new_h = h[idx, :]
#     values = torch.unsqueeze(values, -1)
#     new_h = torch.mul(new_h, values)
#     un_g = g.bool().float()
#     un_g = torch.matmul(un_g, un_g).bool().float()
#     un_g = un_g[idx, :]
#     un_g = un_g[:, idx]
#     g = norm_g(un_g)
#     return g, new_h, idx
#
#
# def norm_g(g):
#     degrees = torch.sum(g, 1)
#     g = g / degrees
#     return g
#
#
# class Initializer(object):
#
#     @classmethod
#     def _glorot_uniform(cls, w):
#         if len(w.size()) == 2:
#             fan_in, fan_out = w.size()
#         elif len(w.size()) == 3:
#             fan_in = w.size()[1] * w.size()[2]
#             fan_out = w.size()[0] * w.size()[2]
#         else:
#             fan_in = np.prod(w.size())
#             fan_out = np.prod(w.size())
#         limit = np.sqrt(6.0 / (fan_in + fan_out))
#         w.uniform_(-limit, limit)
#
#     @classmethod
#     def _param_init(cls, m):
#         if isinstance(m, nn.parameter.Parameter):
#             cls._glorot_uniform(m.data)
#         elif isinstance(m, nn.Linear):
#             m.bias.data.zero_()
#             cls._glorot_uniform(m.weight.data)
#
#     @classmethod
#     def weights_init(cls, m):
#         for p in m.modules():
#             if isinstance(p, nn.ParameterList):
#                 for pp in p:
#                     cls._param_init(pp)
#             else:
#                 cls._param_init(p)
#
#         for name, p in m.named_parameters():
#             if '.' not in name:
#                 cls._param_init(p)

#Fuse U Net
import torch
import torch.nn as nn
import numpy as np


class GraphUnet(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.bottom_gcn = GCN(dim, dim, act, drop_p)
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim, act, drop_p))
            self.up_gcns.append(GCN(dim, dim, act, drop_p))
            self.pools.append(Pool(ks[i], dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))
        self.fuselayer_1=FuseLayer(dim,act,drop_p,self.l_n+1)
        # self.fuselayer_2 = FuseLayer(dim, act, drop_p, self.l_n + 1)
        # self.fuselayer_3 = FuseLayer(dim, act, drop_p, self.l_n + 1)
        # self.fuselayer_4 = FuseLayer(dim, act, drop_p, self.l_n + 1)
        # self.GCNList_1=GCNList(dim,act,drop_p,self.l_n+1)
        # self.GCNList_2 = GCNList(dim, act, drop_p, self.l_n + 1)



    def forward(self, g, h):
        adj_ms = []
        indices_list = []
        down_outs = []
        hs = []
        org_h = h
        org_g=g
        for i in range(self.l_n):
            h = self.down_gcns[i](g, h)
            adj_ms.append(g)
            down_outs.append(h)
            g, h, idx = self.pools[i](g, h)

            indices_list.append(idx)
        adj_ms.append(g)
        down_outs.append(h)

        # down_outs=self.GCNList_1(adj_ms,down_outs)
        down_outs_fuse=self.fuselayer_1(adj_ms,down_outs,indices_list)
        # down_outs = self.fuselayer_2(adj_ms, down_outs, indices_list)
        # down_outs = self.fuselayer_3(adj_ms, down_outs, indices_list)
        # down_outs = self.fuselayer_4(adj_ms, down_outs, indices_list)

        # down_outs = self.GCNList_2(adj_ms, down_outs, indices_list)

        h=down_outs_fuse[self.l_n]
        # h = down_outs[self.l_n]+down_outs_fuse[self.l_n]+self.bottom_gcn(adj_ms[self.l_n], down_outs_fuse[self.l_n])
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            g, idx = adj_ms[up_idx], indices_list[up_idx]
            g, h = self.unpools[i](g, h, down_outs_fuse[up_idx], idx)
            h = self.up_gcns[i](g, h)
            # h = h.add(down_outs[up_idx])
            h=h.add(down_outs_fuse[up_idx])
            hs.append(h)
        h = h.add(org_h)
        hs.append(h)
        return hs




class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, act, p):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

    def forward(self, g, h):
        h = self.drop(h)
        h = torch.matmul(g, h)
        h = self.proj(h)
        h = self.act(h)
        return h


class Pool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)


class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return g, new_h

class Up_One(nn.Module):
    def __init__(self,dim, act, drop_p):
        super(Up_One, self).__init__()
        # self.gcn=GCN(dim,dim,act,drop_p)
    def forward(self,target_g, original_level_h,  original_level_idx):
        new_h = original_level_h.new_zeros([target_g.shape[0], original_level_h.shape[1]])
        new_h[original_level_idx] = original_level_h
        # new_h=self.gcn(target_g,new_h)
        return new_h


class Up(nn.Module):
    def __init__(self,dim, act, drop_p,begin_level,final_level ):
        super(Up, self).__init__()
        # self.gcn=GCN(in_dim, out_dim, act, p)
        self.begin_level=begin_level
        self.final_level=final_level
        proess_list=nn.ModuleList()
        for i in range(begin_level-final_level):
            proess_list.append(Up_One(dim, act, drop_p))
        self.process_list=proess_list

        self.gcn = GCN(dim, dim, act, drop_p)

    def forward(self, adj_ms,down_outs, idx_list):
        # process_h = self.gcn(original_level_g,original_level_h)
        h_value=down_outs[self.begin_level]
        for i in range(self.begin_level-self.final_level):
            h_value=self.process_list[i](adj_ms[self.begin_level-i-1],h_value,idx_list[self.begin_level-i-1])
        h_value=self.gcn(adj_ms[self.final_level],h_value)


        return h_value


class Down_One(nn.Module):
    def __init__(self, in_dim, out_dim, act, p):
        super(Down_One, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        # self.gcn=GCN(in_dim, out_dim, act, p)
    def forward(self, this_level_g, this_level_h,idx):
        # h=self.gcn(this_level_g,this_level_h)
        # Z = self.drop(h)
        # weights = self.proj(Z).squeeze()
        # scores = self.sigmoid(weights)
        # values=scores[idx]
        # new_h = h[idx, :]
        # values = torch.unsqueeze(values, -1)
        # new_h = torch.mul(new_h, values)

        new_h=this_level_h[idx,:]
        return new_h

class Down_Multi(nn.Module):
    def __init__(self, in_dim, out_dim, act, p,begin_level,final_level):
        super(Down_Multi, self).__init__()
        process=nn.ModuleList()
        for i in range(final_level-begin_level):
            process.append(Down_One(in_dim, out_dim, act, p))
        self.process=process
        self.begin_level=begin_level
        self.final_level=final_level
        self.gcn = GCN(in_dim, out_dim, act, p)
    def forward(self, adj_ms,down_outs, idx_list):
        h_value = down_outs[self.begin_level]
        for i in range(self.final_level-self.begin_level):

            h_value = self.process[i](adj_ms[self.begin_level+i ], h_value, idx_list[self.begin_level+i])
            # if (self.final_level == 2 and self.begin_level == 0):
            #     print("H Value Size {}".format(h_value.size()))
        h_value=self.gcn(adj_ms[self.final_level],h_value)
        return h_value


class FuseLayer(nn.Module):
    def __init__(self,dim, act, drop_p,list_length):
        super(FuseLayer, self).__init__()
        self.list_length=list_length
        processList = nn.ModuleList()
        for i in range(self.list_length):
            temp = nn.ModuleList()
            for j in range(self.list_length):
                if i == j:
                    temp.append(GCN(dim, dim, act, drop_p))
                elif i < j:
                    temp.append(Up(dim,act,drop_p,j, i))
                else:
                    temp.append(Down_Multi(dim, dim, act, drop_p, j, i))
            processList.append(temp)
        self.processList = processList
    def forward(self,adj_ms,down_outs, idx_list):
        result=[]
        for i in range(self.list_length):
            temp=0
            for j in range(self.list_length):
                if i==j:
                    calculate=self.processList[i][j](adj_ms[i],down_outs[i])
                    # print("Directly From {} to {} result size {}".format(j,i,calculate.size()))
                    temp+=calculate
                else:
                    calculate = self.processList[i][j](adj_ms, down_outs,idx_list)
                    # print("From {} to {} result size {}".format(j,i,calculate.size()))
                    temp += calculate
            result.append(temp)
        return result
class GCNList(nn.Module):
    def __init__(self,dim, act, drop_p,list_length):
        super(GCNList, self).__init__()
        process=nn.ModuleList()
        self.list_length=list_length
        for i in range(list_length):
            process.append(GCN(dim, dim, act, drop_p))
        self.process=process
    def forward(self,adj_ms,down_outs):
        temp_result=[]
        for i in range(self.list_length):
            temp_result.append(self.process[i](adj_ms[i],down_outs[i]))
        return temp_result

def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx


def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g


class Initializer(object):

    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)



