interface BadgeProps {
  children: React.ReactNode;
  variant?: 'default' | 'success' | 'danger' | 'warning' | 'info' | 'neutral';
  className?: string;
}

const variantClasses: Record<string, string> = {
  default: 'bg-accent/20 text-accent',
  success: 'bg-bull/20 text-bull',
  danger: 'bg-bear/20 text-bear',
  warning: 'bg-yellow-500/20 text-yellow-400',
  info: 'bg-blue-500/20 text-blue-400',
  neutral: 'bg-gray-500/20 text-gray-400',
};

export default function Badge({ children, variant = 'default', className = '' }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 text-xs font-medium rounded-sm ${variantClasses[variant]} ${className}`}
    >
      {children}
    </span>
  );
}
