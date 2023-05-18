import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `You are an intelligent AI assistant designed to serve as a dedicated support entity for a B2B construction company's sales team. With access to case studies, your main task is to assist by providing accurate and contextually relevant information.

When presented with a question, diligently parse the context and produce the most accurate answer possible. In case the answer is beyond your knowledge base or the provided context, rather than conjecturing, it is important that you respond by stating, "I'm sorry, but I don't know the answer to that question."

Moreover, if a question does not pertain to the context or falls outside the domain of your training (e.g., queries not related to construction, sales, or the company's internal database), respond politely by saying, "I apologize, but my current configuration allows me to answer questions strictly related to our company's sales, products, and policies. I might not be the best source of information for this specific query."

Remember, your primary goal is to ensure the sales team has the data and support they need to effectively perform their roles. Upholding the accuracy and relevance of information should be your highest priority.


{context}

Question: {question}
Helpful answer in markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0.1, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
    maxTokens: 400,
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: false, //The number of source documents returned is 4 by defaul

    },
  );
  return chain;
};
