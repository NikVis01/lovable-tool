import React from 'react'
import { ChatBox } from './components/ChatBox'
import { CandidatesTable } from './components/CandidatesTable'

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <div className="max-w-5xl mx-auto px-4 py-6 space-y-8">
        <header className="flex items-center justify-between">
          <h1 className="text-2xl font-semibold">Langflow Chat</h1>
        </header>
        <main className="grid md:grid-cols-2 gap-6">
          <section className="bg-white rounded-lg shadow p-4">
            <h2 className="font-medium mb-2">Chat</h2>
            <ChatBox />
          </section>
          <section className="bg-white rounded-lg shadow p-4">
            <h2 className="font-medium mb-2">Candidates</h2>
            <CandidatesTable />
          </section>
        </main>
      </div>
    </div>
  )
}
