import torch

from models.gpt2 import GPT2Model

from transformers import GPT2Model as OpenAIGPT2Model
from utils import model_size_to_params




def test_gpt2(model_size='gpt2'):
  sent_ids = torch.tensor([[101, 7592, 2088, 102, 0, 0, 0, 0],
                           [101, 7592, 15756, 2897, 2005, 17953, 2361, 102]])
  att_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])

  # Load both the OpenAI and your own model.
  openai_model = OpenAIGPT2Model.from_pretrained(model_size)
  gpt = GPT2Model.from_pretrained(model=model_size, **model_size_to_params(model_size))

  outputs = gpt(sent_ids, att_mask)
  openai_outputs = openai_model(input_ids=sent_ids, attention_mask=att_mask, output_hidden_states=True).hidden_states[-1]

  att_mask = att_mask.unsqueeze(-1)
  outputs['last_hidden_state'] = outputs['last_hidden_state'] * att_mask
  openai_outputs *= att_mask
 
  print("\n🔹 Your GPT-2 Output Shape:", outputs['last_hidden_state'].shape)
  print("🔹 OpenAI GPT-2 Output Shape:", openai_outputs.shape)
    
  print("\n🔹 Your GPT-2 Output (first token, first 5 values):", outputs['last_hidden_state'][0, 0, :5])
  print("🔹 OpenAI GPT-2 Output (first token, first 5 values):", openai_outputs[0, 0, :5])
  
  # check absolute difference
  diff = torch.abs(outputs['last_hidden_state'] - openai_outputs).max()
  print("\n⚠️ Max absolute difference:", diff.item())
  
  assert torch.allclose(outputs['last_hidden_state'], openai_outputs, atol=1e-1, rtol=1e-2)
  


  print("Your GPT2 implementation is correct!")

if __name__ == '__main__':
  test_gpt2('gpt2')
