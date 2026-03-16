interface CardProps {
  children: React.ReactNode;
  className?: string;
  title?: string;
  action?: React.ReactNode;
}

export default function Card({ children, className = '', title, action }: CardProps) {
  return (
    <div className={`bg-dark-card border border-dark-border rounded-md ${className}`}>
      {title && (
        <div className="flex items-center justify-between px-4 py-3 border-b border-dark-border">
          <h3 className="text-sm font-semibold text-gray-200">{title}</h3>
          {action}
        </div>
      )}
      {children}
    </div>
  );
}
