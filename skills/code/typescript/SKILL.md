# SKILL: code/typescript — TypeScript / React Implementation

## Purpose
Guides implementation agents producing TypeScript or React artifacts.

## Environment constraints
- TypeScript strict mode. No `any` unless absolutely unavoidable with a comment.
- React 18+. Functional components only — no class components.
- Tailwind CSS for styling (core utility classes only — no custom config required).
- No external state libraries (Zustand, Redux) unless explicitly listed in requirements.
- Fetch API or React Query for data fetching — no axios unless already in deps.

## Code quality bar
- Every component must have a JSDoc comment describing its purpose and props.
- Props interfaces must be defined above each component, not inline.
- No magic strings for API endpoints — use named constants.
- Loading, error, and empty states must be handled explicitly.
- Accessibility: interactive elements need aria labels; images need alt text.

## Output format
Produce a single complete `.tsx` or `.ts` file per artifact. Structure:
```
// imports
// types / interfaces
// constants
// component / function
// export
```

## What to include in findings
- Component/function names and their responsibilities.
- State shape decisions and why.
- API contracts assumed (endpoint shapes, response types).
- Accessibility considerations addressed.

## Common failure modes — avoid these
- Components with no error boundary or loading state.
- Hardcoded data that should come from props or API.
- Missing TypeScript types on function parameters or return values.
- Using `useEffect` with missing or incorrect dependency arrays.
