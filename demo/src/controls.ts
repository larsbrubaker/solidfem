// The control builders the page uses (segmented choice, slider, dropdown), so the page shares one
// look; styles/main.css styles them. A slider sits in a wrapper that takes touch input in its
// place (src/slider-touch.ts), so a finger scrolling the page past it does not move it. A slider
// can be re-ranged after creation, because the allowed ranges depend on the chosen combination,
// and the DOM is never rebuilt so keyboard focus stays put.

import { attachTouchDrag } from './slider-touch.ts';

export interface SliderSpec {
  /** Position-space range and step of the native input. */
  min: number;
  max: number;
  step: number;
  /** The value at a position, already inside the combination's range. */
  toValue(position: number): number;
  /** The position nearest to a value. */
  toPosition(value: number): number;
  /** The value as text beside the label. */
  format(value: number): string;
}

export interface Slider {
  element: HTMLElement;
  update(spec: SliderSpec, value: number): void;
}

export function createSlider(label: string, spec: SliderSpec, value: number, onChange: (value: number) => void): Slider {
  const group = document.createElement('div');
  group.className = 'control-group';
  const lbl = document.createElement('label');
  const labelText = document.createElement('span');
  labelText.textContent = label;
  const valueText = document.createElement('span');
  valueText.className = 'slider-value';
  lbl.appendChild(labelText);
  lbl.appendChild(valueText);

  const input = document.createElement('input');
  input.type = 'range';
  input.setAttribute('aria-label', label);

  let current = spec;
  let currentValue = value;
  const showValue = () => {
    const text = current.format(currentValue);
    valueText.textContent = text;
    input.setAttribute('aria-valuetext', text);
  };
  input.addEventListener('input', () => {
    currentValue = current.toValue(parseFloat(input.value));
    showValue();
    onChange(currentValue);
  });

  const hold = document.createElement('div');
  hold.className = 'slider-hold';
  hold.appendChild(input);
  attachTouchDrag(hold, input);
  group.appendChild(lbl);
  group.appendChild(hold);

  const update = (next: SliderSpec, nextValue: number) => {
    current = next;
    currentValue = nextValue;
    // The bounds go first: a browser clamps value to whichever bounds are current.
    input.min = String(next.min);
    input.max = String(next.max);
    input.step = String(next.step);
    input.value = String(next.toPosition(nextValue));
    showValue();
  };
  update(spec, value);
  return { element: group, update };
}

export interface Segmented {
  element: HTMLElement;
  set(value: string): void;
}

export function createSegmented(
  label: string, options: { value: string; text: string }[], selected: string,
  onChange: (value: string) => void,
): Segmented {
  const group = document.createElement('div');
  group.className = 'control-group';
  const lbl = document.createElement('label');
  lbl.textContent = label;
  group.appendChild(lbl);
  const row = document.createElement('div');
  row.className = 'segmented';
  row.setAttribute('role', 'group');
  row.setAttribute('aria-label', label);
  const buttons = new Map<string, HTMLButtonElement>();
  const show = (value: string) => {
    for (const [key, button] of buttons) {
      const on = key === value;
      button.classList.toggle('active', on);
      button.setAttribute('aria-pressed', String(on));
    }
  };
  for (const option of options) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = option.text;
    button.addEventListener('click', () => {
      show(option.value);
      onChange(option.value);
    });
    buttons.set(option.value, button);
    row.appendChild(button);
  }
  group.appendChild(row);
  show(selected);
  return { element: group, set: show };
}

export function createDropdown(
  label: string, options: { value: string; text: string }[], selected: string,
  onChange: (value: string) => void,
): HTMLElement {
  const group = document.createElement('div');
  group.className = 'control-group';
  const lbl = document.createElement('label');
  lbl.textContent = label;
  const select = document.createElement('select');
  select.setAttribute('aria-label', label);
  for (const option of options) {
    const o = document.createElement('option');
    o.value = option.value;
    o.textContent = option.text;
    select.appendChild(o);
  }
  select.value = selected;
  select.addEventListener('change', () => onChange(select.value));
  group.appendChild(lbl);
  group.appendChild(select);
  return group;
}
