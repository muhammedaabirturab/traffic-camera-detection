import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { api } from './api.js'

const StatusContext = createContext({ health: null, online: false, refresh: () => {} })

// Polls /api/health so every page knows which detectors are loaded.
export function StatusProvider({ children }) {
  const [health, setHealth] = useState(null)
  const [online, setOnline] = useState(false)

  const refresh = useCallback(async () => {
    try {
      setHealth(await api.health())
      setOnline(true)
    } catch {
      setOnline(false)
    }
  }, [])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, health?.models_loaded ? 15000 : 3000)
    return () => clearInterval(id)
  }, [refresh, health?.models_loaded])

  return <StatusContext.Provider value={{ health, online, refresh }}>{children}</StatusContext.Provider>
}

export const useStatus = () => useContext(StatusContext)
