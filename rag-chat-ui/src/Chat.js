import React, { useState, useEffect, useRef } from "react";

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;
    

    const userMessage = { from: "user", text: input, time: new Date().toLocaleTimeString() };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const res = await fetch("http://localhost:8000/generate", { 
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: input }),
      });
      const data = await res.json();

      const llmMessage = { from: "llm", text: data.response || "Failed to receive a response", time: new Date().toLocaleTimeString() };
      setMessages(prev => [...prev, llmMessage]);
    } catch {
      const errorMessage = { from: "llm", text: "Failed to connect to the server", time: new Date().toLocaleTimeString() };
      setMessages(prev => [...prev, errorMessage]);
    }
    finally{
        setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen max-w-xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-white shadow-lg rounded-lg">
  <h1 className="text-2xl font-bold mb-4 text-center text-blue-700">Smart Chat</h1>

  <div className="flex-1 overflow-y-auto space-y-4 p-4 bg-white rounded-lg shadow-inner flex flex-col">
    {messages.length === 0 && (
      <p className="text-gray-400 text-center mt-10">Start a Conversation ...</p>
    )}

    {messages.map((msg, idx) => (
      <div
        key={idx}
        className={`max-w-[70%] p-4 rounded-2xl shadow
          ${msg.from === "user"
            ? "bg-blue-600 text-white self-end rounded-br-none"
            : "bg-gray-200 text-gray-800 self-start rounded-bl-none"
          }`}
      >
        <p className="whitespace-pre-wrap">{msg.text}</p>
        <div className="text-xs mt-2 text-gray-400 text-left">{msg.time}</div>
      </div>
    ))}

    {isLoading && (
      <div className="max-w-[70%] p-4 rounded-2xl shadow bg-gray-100 text-gray-600 self-start italic animate-pulse">
        AI is typing...
      </div>
    )}

    <div ref={messagesEndRef} />
  </div>

      <div className="mt-4 flex">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          className="flex-1 p-3 border border-blue-300 rounded-l-2xl focus:outline-none focus:ring-2 focus:ring-blue-400"
          placeholder="write your message ...."
          onKeyDown={e => e.key === "Enter" && sendMessage()}
        />
        <button
          onClick={sendMessage}
          className="bg-blue-600 hover:bg-blue-700 transition-colors text-white px-6 rounded-r-2xl"
        >
          Send
        </button>
      </div>
    </div>
  );
}
