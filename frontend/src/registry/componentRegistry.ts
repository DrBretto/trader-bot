// PKT-TB-BV-01 — the declarative component registry, typed for the frontend.
//
// SINGLE SOURCE OF TRUTH for what components the brain has and how each is
// measured. Both the nightly contribution writer (Python) and this frontend
// render OFF the same `component_registry.json`. It carries NO measured numbers
// — existence + how-it-would-be-measured only; numbers join from the
// contribution ledger (BV-03) on the stable `id`.
//
// No render module may hard-code a component-id array or a component→label map
// (that brittleness is what this registry replaces, and the no-hard-coding CI in
// tests/test_component_registry_no_hardcoding.py fails the build on it). Iterate
// the registry; join the ledger on the stable `id`; the honest-state fallback is
// the universal default for any component with no data.

import registryData from './component_registry.json';

export type ExpectedStatus = 'live' | 'comparison' | 'retired' | 'planned';
export type InstrumentationStatus =
  | 'wired' | 'partial' | 'awaiting_data' | 'not_wired' | 'retired';

export interface Attribution {
  method: string;
  book: string | null;
  book_pair: string | null;
  denominator: string | null;
  strippable: boolean;
  non_strip_reason: string | null;
}

export interface ComponentDescriptor {
  id: string;                 // stable join key — never display text, never reused
  label: string;              // the ONLY display string
  blurb: string;
  group: string;
  kind: string;
  parent: string | null;
  order: number;
  attribution: Attribution;
  expected_status: ExpectedStatus;
  instrumentation_status: InstrumentationStatus;
  instrumentation_note: string;
  history_of: string[];
}

export interface LadderBook {
  book: string;
  parent: string | null;
  component: string | null;   // null = the incumbent baseline book (no organ)
  history_of: string[];
  note?: string;
}

export interface Registry {
  schema: string;
  note?: string;
  id_rules?: string;
  components: ComponentDescriptor[];
  ladder_books: LadderBook[];
}

export const registry = registryData as unknown as Registry;

/** All descriptors, sorted by descriptor.order (the general Brain-Map order). */
export function componentsInOrder(): ComponentDescriptor[] {
  return [...registry.components].sort((a, b) => a.order - b.order);
}

const BY_ID: Map<string, ComponentDescriptor> = new Map(
  registry.components.map((c) => [c.id, c]),
);

/** Descriptor for a stable id, or undefined (caller renders the honest fallback). */
export function byId(id: string): ComponentDescriptor | undefined {
  return BY_ID.get(id);
}

/** One rent-ladder rung, joined from the registry (never hard-coded). */
export interface LadderRung {
  id: string;                 // stable id (the ledger join key)
  label: string;
  book_pair: string;          // e.g. "F-R", from the descriptor's attribution
  descriptor: ComponentDescriptor;
}

/**
 * The rent-ladder rungs, in ladder order (the canonical I→R→F→E→U sequence of
 * `ladder_books`), with the baseline incumbent book (component === null)
 * skipped. The RentLedger maps over this — no LADDER_RUNGS / COMPONENT_LABEL.
 */
export function ladderRungs(): LadderRung[] {
  const out: LadderRung[] = [];
  for (const lb of registry.ladder_books) {
    if (!lb.component) continue;               // baseline book — not a rung
    const d = BY_ID.get(lb.component);
    if (!d) continue;                          // missing descriptor → MISSING_COMPONENT (BV-04)
    out.push({
      id: d.id,
      label: d.label,
      book_pair: d.attribution.book_pair ?? '',
      descriptor: d,
    });
  }
  return out;
}

/** The forecast (M1) rent rung's stable id, read from the ladder (book "F").
 *  Exported so consumers never hard-code the literal 'forecast'. */
export const FORECAST_RUNG_ID: string =
  registry.ladder_books.find((b) => b.book === 'F')?.component ?? 'forecast';
