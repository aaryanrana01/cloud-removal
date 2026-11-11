import { useCallback, useRef, useState } from 'react'

/**
 * Drag the handle (or use arrow keys) to reveal `before` on the left and `after` on the right.
 * The `after` image sits in normal flow and sets the size; `before` is layered on top and clipped.
 */
export default function CompareSlider({ before, after, beforeLabel, afterLabel }) {
  const [pos, setPos] = useState(50)
  const box = useRef(null)
  const dragging = useRef(false)

  const moveTo = useCallback((clientX) => {
    const r = box.current.getBoundingClientRect()
    setPos(Math.min(100, Math.max(0, ((clientX - r.left) / r.width) * 100)))
  }, [])

  const onKey = (e) => {
    const step = e.shiftKey ? 10 : 2
    const next = { ArrowLeft: pos - step, ArrowRight: pos + step, Home: 0, End: 100 }[e.key]
    if (next === undefined) return
    e.preventDefault()
    setPos(Math.min(100, Math.max(0, next)))
  }

  return (
    <div
      ref={box}
      className="relative select-none touch-none cursor-ew-resize overflow-hidden bg-ink"
      onPointerDown={(e) => { dragging.current = true; e.currentTarget.setPointerCapture(e.pointerId); moveTo(e.clientX) }}
      onPointerMove={(e) => dragging.current && moveTo(e.clientX)}
      onPointerUp={() => { dragging.current = false }}
      onPointerCancel={() => { dragging.current = false }}
    >
      <img src={after} alt={afterLabel} className="block w-full h-auto" draggable="false" />
      <img
        src={before}
        alt={beforeLabel}
        className="absolute inset-0 w-full h-full object-cover"
        style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
        draggable="false"
      />

      <Tag side="left">{beforeLabel}</Tag>
      <Tag side="right">{afterLabel}</Tag>

      <div className="absolute inset-y-0 w-0.5 bg-white shadow-[0_0_0_1px_rgba(22,38,46,.35)]" style={{ left: `${pos}%` }}>
        <div
          role="slider"
          tabIndex={0}
          aria-label={`Compare ${beforeLabel} and ${afterLabel}`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(pos)}
          onKeyDown={onKey}
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 grid place-items-center
                     size-11 rounded-full bg-white text-ink shadow-[0_2px_10px_rgba(22,38,46,.35)]"
        >
          <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
            <path d="M8 6 3 11l5 5M14 6l5 5-5 5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      </div>
    </div>
  )
}

function Tag({ side, children }) {
  return (
    <span
      className={`pointer-events-none absolute top-3 ${side === 'left' ? 'left-3' : 'right-3'}
                  rounded-sm bg-ink/75 px-2 py-1 text-[0.8125rem] font-medium text-white backdrop-blur-sm`}
    >
      {children}
    </span>
  )
}
