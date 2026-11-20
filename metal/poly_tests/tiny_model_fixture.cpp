// Synthetic two-layer topologies for engine smoke checks, not language-model
// weights or quality evidence. No network or external model assets are used.
// Puddle's Metal strata-poly runs qwen35moe (the default; qwen35moe-fp16 has heads of 48) and gemma4.
#include "ggml.h"
#include "gguf.h"
#include <cmath>
#include <string>
#include <vector>
int main(int argc,char** argv){
    if(argc<2 || argc>3)return 2;
    auto* ctx=ggml_init({32*1024*1024,nullptr,false});auto* file=gguf_init_empty();
    const std::string requested=argc==3?argv[2]:"qwen35moe";
    const bool fp16_only=requested=="qwen35moe-fp16";
    const std::string arch=fp16_only?"qwen35moe":requested;
    const int head_dim=fp16_only?48:64;
    const bool qwen=arch=="qwen35moe",gemma=arch=="gemma4";
    if(!qwen && !gemma)return 2;
    gguf_set_val_str(file,"general.architecture",arch.c_str());
    auto integer=[&](const char* key,int v){gguf_set_val_u32(file,(arch+"."+key).c_str(),v);};
    auto real=[&](const char* key,float v){gguf_set_val_f32(file,(arch+"."+key).c_str(),v);};
    integer("block_count",qwen?3:2);if(qwen)integer("nextn_predict_layers",1);integer("embedding_length",256);integer("vocab_size",256);integer("expert_count",4);
    integer("expert_used_count",2);integer("expert_feed_forward_length",256);integer("attention.head_count",4);
    integer("expert_gating_func",2);real("attention.layer_norm_rms_epsilon",1e-5f);real("rope.freq_base",10000);
    integer("attention.head_count_kv",1);integer("attention.key_length",head_dim);integer("attention.value_length",head_dim);
    if(qwen){integer("ssm.state_size",64);integer("ssm.group_count",1);integer("ssm.time_step_rank",2);integer("ssm.inner_size",128);integer("ssm.conv_kernel",4);
        uint32_t section=head_dim/4,sections[]={section,section,section};gguf_set_arr_data(file,"qwen35moe.rope.dimension_sections",GGUF_TYPE_UINT32,sections,3);}
    else {integer("attention.sliding_window",7);integer("attention.key_length_swa",64);integer("attention.value_length_swa",64);
        uint32_t pattern[]={1,0};gguf_set_arr_data(file,"gemma4.attention.sliding_window_pattern",GGUF_TYPE_UINT32,pattern,2);real("final_logit_softcapping",30);}
    auto tensor=[&](std::string name,std::vector<int64_t> shape,bool quant=false,bool norm=false){
        auto* t=ggml_new_tensor(ctx,quant?GGML_TYPE_Q8_0:GGML_TYPE_F32,shape.size(),shape.data());ggml_set_name(t,name.c_str());
        std::vector<float> values(ggml_nelements(t));
        for(size_t i=0;i<values.size();++i)values[i]=norm?1.f:.03f*std::sin(float(i)*.079f+float(name.size()));
        if(quant)ggml_quantize_chunk(GGML_TYPE_Q8_0,values.data(),t->data,0,values.size()/shape[0],shape[0],nullptr);
        else std::copy(values.begin(),values.end(),static_cast<float*>(t->data));
        gguf_add_tensor(file,t);
    };
    tensor("token_embd.weight",{256,256},true);tensor("output_norm.weight",{256},false,true);
    if(gemma)tensor("rope_freqs.weight",{32},false,true);tensor("output.weight",{256,256},true);
    for(int layer=0;layer<(qwen?3:2);++layer){std::string p="blk."+std::to_string(layer)+".";
        tensor(p+"attn_norm.weight",{256},false,true);tensor(p+(qwen?"post_attention_norm.weight":"ffn_norm.weight"),{256},false,true);
        if(layer==0 && qwen){tensor(p+"attn_qkv.weight",{256,256},true);tensor(p+"attn_gate.weight",{256,128},true);
            tensor(p+"ssm_conv1d.weight",{4,256});tensor(p+"ssm_dt.bias",{2});tensor(p+"ssm_a",{2});tensor(p+"ssm_alpha.weight",{256,2});tensor(p+"ssm_beta.weight",{256,2});tensor(p+"ssm_norm.weight",{64},false,true);tensor(p+"ssm_out.weight",{128,256},true);}
        else{tensor(p+"attn_q.weight",{256,qwen?8*head_dim:256},true);tensor(p+"attn_k.weight",{256,head_dim},true);tensor(p+"attn_v.weight",{256,head_dim},true);tensor(p+"attn_output.weight",{4*head_dim,256},true);tensor(p+"attn_q_norm.weight",{head_dim},false,true);tensor(p+"attn_k_norm.weight",{head_dim},false,true);}
        if(qwen){tensor(p+"ffn_gate_shexp.weight",{256,256},true);tensor(p+"ffn_up_shexp.weight",{256,256},true);tensor(p+"ffn_down_shexp.weight",{256,256},true);tensor(p+"ffn_gate_inp_shexp.weight",{256});}
        if(gemma){for(const char* name:{"post_attention_norm.weight","post_ffw_norm.weight","ffn_gate_inp.scale","post_ffw_norm_1.weight","pre_ffw_norm_2.weight","post_ffw_norm_2.weight"})tensor(p+name,{256},false,true);
            tensor(p+"ffn_gate.weight",{256,256},true);tensor(p+"ffn_up.weight",{256,256},true);tensor(p+"ffn_down.weight",{256,256},true);tensor(p+"ffn_down_exps.scale",{4},false,true);}
        tensor(p+"ffn_gate_inp.weight",{256,4});tensor(p+"ffn_gate_up_exps.weight",{256,512,4},true);tensor(p+"ffn_down_exps.weight",{256,256,4},true);tensor(p+"exp_probs_b.bias",{4});
    }
    if(qwen){tensor("blk.2.nextn.eh_proj.weight",{512,256},true);tensor("blk.2.nextn.enorm.weight",{256},false,true);tensor("blk.2.nextn.hnorm.weight",{256},false,true);}
    bool ok=gguf_write_to_file(file,argv[1],false);gguf_free(file);ggml_free(ctx);return ok?0:1;
}
