// Small wrappers around the FastAPI endpoints. Every error becomes a readable message.

async function request(url, options) {
  let res
  try {
    res = await fetch(url, options)
  } catch {
    throw new Error('Could not reach the server. Check your connection and try again.')
  }
  const isJson = res.headers.get('content-type')?.includes('application/json')
  const body = isJson ? await res.json() : null
  if (!res.ok) {
    if (body?.detail) throw new Error(typeof body.detail === 'string' ? body.detail : 'The request was rejected.')
    if (res.status === 502 || res.status === 503) throw new Error('The server is starting up. Try again in about a minute.')
    throw new Error(`The server returned an error (${res.status}).`)
  }
  return body
}

export const getSamples = () => request('/api/samples')
export const getModelInfo = () => request('/api/model')

export function predict(file) {
  const form = new FormData()
  form.append('file', file)
  return request('/api/predict', { method: 'POST', body: form })
}
