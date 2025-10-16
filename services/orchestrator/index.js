import { InferenceClient } from '@huggingface/inference';
import express from 'express';
import axios from 'axios';
import cors from 'cors';
import dotenv from 'dotenv';
// dotenv.config({ path: '../../.env', debug: true });
dotenv.config();

const app = express();
app.use(express.json());
app.use(cors());

const HF_TOKEN = process.env.HF_API_KEY;
console.log(HF_TOKEN);
const EMBEDDING_URL = process.env.EMBEDDING_URL;
console.log('hi:',EMBEDDING_URL);
const client = new InferenceClient(HF_TOKEN);

app.post('/chat', async (req, res) => {
  const { user_id, query, k = 5 } = req.body;
  if (!user_id || !query) {
    return res.status(400).json({ error: 'user_id and query are required' });
  }
  const startTime = Date.now(); // Start timing
  try {
    // 1. Call embedding service for vector search
    console.log(EMBEDDING_URL);
    const searchResp = await axios
  .post(`${process.env.EMBEDDING_URL}/search`, { query, k })
  .catch((err) => {
    console.error('Axios error:', err);
    throw err; // Re-throw to be caught by the outer try-catch
  });

   // const searchResp = await axios.post(`${EMBEDDING_URL}/search`, { query, k });
    console.log('Search response:', searchResp.data);
    if (!searchResp.data || !searchResp.data.results) {
      return res.status(500).json({ error: 'Invalid response from embedding service' });
    }
    const results = searchResp.data.results;

    // 2. Construct context
    const context = results.map((r, i) => `(${i + 1}) ${r.text}`).join('\n');

    // 3. Build prompt for LLM
    const prompt = `You are a helpful assistant. Use the following sources to answer:\n\n${context}\n\nQuestion: ${query}\nAnswer:`;

    // 4. Call Hugging Face Inference API (lightweight model)
    const hfResp = await client.chatCompletion({
      provider: 'together',
      model: 'deepseek-ai/DeepSeek-V3.1',
      messages: [{ role: 'user', content: prompt }],
      stream: false,
    });

    const llmAnswer = hfResp.choices?.[0]?.message?.content || 'No answer generated.';
    const endTime = Date.now(); // End timing
    const timing_ms = endTime - startTime;
    
    // 5. Return response
    res.json({ user_id, answer: llmAnswer, source_docs: results, timing_ms: timing_ms });
  } catch (err) {
    console.error('Error details:', err.response ? err.response.data : err.message);
    res.status(600).json({ error: err.response ? err.response.data : err.message });
  }
});
const PORT = process.env.PORT;
console.log(`Starting server on port ${PORT}`);
app.listen(PORT, () => {
  console.log(`Orchestrator service running on port ${PORT}`);
});