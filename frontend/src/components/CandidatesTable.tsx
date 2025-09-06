import React, { useEffect, useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

type Row = Record<string, any>

export function CandidatesTable() {
  const [rows, setRows] = useState<Row[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`${API_URL}/api/candidates`)
        const data = await r.json()
        setRows(data.rows ?? [])
      } catch (e: any) {
        setError('Failed to load candidates')
      }
    })()
  }, [])

  if (error) return <div className="text-sm text-red-600">{error}</div>
  if (!rows.length) return <div className="text-sm text-gray-500">No rows</div>

  const headers = Object.keys(rows[0])

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="bg-gray-100">
            {headers.map(h => (
              <th key={h} className="text-left px-3 py-2 font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t">
              {headers.map(h => (
                <td key={h} className="px-3 py-2 align-top whitespace-pre-wrap">{String(r[h] ?? '')}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
