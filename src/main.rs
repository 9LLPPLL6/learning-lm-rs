mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use tokenizers::Tokenizer;
use std::io::{self, Write};

fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    // let input = "Once upon a time";
    // let binding = tokenizer.encode(input, true).unwrap();
    // let input_ids = binding.get_ids();
    // print!("\n{}\n", input);
    // let output_ids = llama.generate(
    //     input_ids,
    //     500,
    //     0.9,
    //     4,
    //     1.,
    // );
    // println!("{}", tokenizer.decode(&output_ids, true).unwrap());
    println!("AI chatbot is ready. Type your message and press Enter.");
    println!("Press exit to quit\n");
    std::io::stdout().flush().unwrap();

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let input_trimmed = input.trim();
    if input_trimmed == "exit" {
        return;
    }
    let chat_input = "<|im_start|>system".to_string() + "\n" + "You are a highly knowledgeable and friendly assistant. Your goal is to understand and respond to user inquiries with clarity. Your interactions are always respectful, helpful, and focused on delivering the most accurate information to the user." + "<|im_end|>" + "\n" + "<|im_start|>user" + "\n"  + input_trimmed +  "<|im_end|>" + "\n" + "<|im_start|>assistant\n";
    let binding = tokenizer.encode(chat_input.as_str(), true).unwrap();
    let input_ids = binding.get_ids();
    let mut cache = llama.new_cache();
    let mut len = 0;
    let output_ids = llama.generate(
        input_ids,
        256,
        0.55,
        35,
        0.65,
        &mut cache,
        &mut len,
    );
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
    // println!("output_ids:{:?}",output_ids);

    std::io::stdout().flush().unwrap();
    
    loop {
        if len >= 256 {
            println!("The conversation has reached the maximum length. \n");
            return;
        }

        println!("Type your message and press Enter.\n");
        std::io::stdout().flush().unwrap();
        input.clear();
        std::io::stdin().read_line(&mut input).unwrap();
        let input_trimmed = input.trim();
        if input_trimmed == "exit" {
            break;
        }
        let chat_input = "<|im_start|>user".to_string() + "\n" + input_trimmed + "<|im_end|>" + "\n" + "<|im_start|>assistant\n";
        let binding = tokenizer.encode(chat_input.as_str(), true).unwrap();
        let input_ids = binding.get_ids();
        let output_ids = llama.generate(
            input_ids,
            256,
            0.55,
            35,
            0.65,
            &mut cache,
            &mut len,
        );
        // println!("output_ids:{:?}",output_ids);
        println!("{}", tokenizer.decode(&output_ids, true).unwrap());
        std::io::stdout().flush().unwrap();
    }
}

#[test]
fn test() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    println!("bos:{} eos:{}",llama.get_bos_token_id(),llama.get_eos_token_id());
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = "<|im_start|><|im_end|>";
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    println!("input_ids:{:?}",input_ids);
}

#[test]
fn test_chat() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = "<|im_start|>system\nYou are a highly knowledgeable and friendly assistant. Your goal is to understand and respond to user inquiries with clarity. Your interactions are always respectful helpful, and focused on delivering the most accurate information to the user.<|im_end|>\n<|im_start|>user\nHey! Got a question for you!<|im_end|>\n<|im_start|>assistant\nSure! What's it?<|im_end|>\n<|im_start|>user\nWhat are some potential applications for quantum computing?<|im_end|>\n<|im_start|>assistant";
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    println!("input_ids_len:{}",input_ids.len());
    let mut cache = llama.new_cache();
    let mut len = 0;
    let output_ids = llama.generate(
        input_ids,
        256,
        0.55,
        35,
        0.65,
        &mut cache,
        &mut len,
    );
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}
