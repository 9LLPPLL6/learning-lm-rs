use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::{SafeTensors, View};
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, ) model.norm.weight
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        //todo!("实现从safetensors文件的模型参数加载");
        let get_tensor = |name: &str| {
            let tensor = safetensor.tensor(name).unwrap();
            let shape = tensor.shape().to_vec();
            let length = shape.iter().fold(1, |acc, &x| acc * x);
            let data = unsafe {
                std::slice::from_raw_parts(tensor.data().as_ptr() as *const f32, tensor.data_len()/4)
            };
            Tensor::new(data.to_vec(), &shape)
        };
        
        let mut rms_att_w = Vec::new();
        let mut wq = Vec::new();
        let mut wk = Vec::new();
        let mut wv = Vec::new();
        let mut wo = Vec::new();
        let mut rms_ffn_w = Vec::new();
        let mut w_up = Vec::new();
        let mut w_gate = Vec::new();
        let mut w_down = Vec::new();

        for i in 0..config.num_hidden_layers {
            rms_att_w.push(get_tensor(&format!("model.layers.{}.input_layernorm.weight", i)));
            wq.push(get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", i)));
            wk.push(get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", i)));
            wv.push(get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", i)));
            wo.push(get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", i)));
            rms_ffn_w.push(get_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", i)));
            w_up.push(get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", i)));
            w_gate.push(get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", i)));
            w_down.push(get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", i)));
        }

        if config.tie_word_embeddings {
            LLamaParams {
                embedding_table: get_tensor("lm_head.weight"),
                rms_att_w: rms_att_w,
                wq: wq,
                wk: wk,
                wv: wv,
                wo: wo,
                rms_ffn_w: rms_ffn_w,
                w_up: w_up,
                w_gate: w_gate,
                w_down: w_down,
                rms_out_w: get_tensor("model.norm.weight"),
                lm_head: get_tensor("lm_head.weight")
            }
        } else {
            LLamaParams {
                embedding_table: get_tensor("model.embed_tokens.weight"),
                rms_att_w: rms_att_w,
                wq: wq,
                wk: wk,
                wv: wv,
                wo: wo,
                rms_ffn_w: rms_ffn_w,
                w_up: w_up,
                w_gate: w_gate,
                w_down: w_down,
                rms_out_w: get_tensor("model.norm.weight"),
                lm_head: get_tensor("lm_head.weight")
            }
        }
    }
}
