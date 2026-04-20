export function ThemeScope({ value, children, className = "" }) {
  return (
    <div data-theme={value} className={className}>
      {children}
    </div>
  );
}
