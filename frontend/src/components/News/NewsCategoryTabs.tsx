const categories = ['All', 'Earnings', 'Policy', 'Product', 'Market', 'Competition', 'Management'];

interface Props {
  active: string;
  onChange: (category: string) => void;
}

export default function NewsCategoryTabs({ active, onChange }: Props) {
  return (
    <div className="flex items-center gap-1 overflow-x-auto pb-1">
      {categories.map((cat) => {
        const isActive = active === cat;
        return (
          <button
            key={cat}
            onClick={() => onChange(cat)}
            className={`px-3 py-1.5 text-xs font-medium rounded-sm whitespace-nowrap transition-colors ${
              isActive
                ? 'bg-accent text-white'
                : 'bg-dark-card text-gray-400 hover:text-gray-200 hover:bg-dark-hover border border-dark-border'
            }`}
          >
            {cat}
          </button>
        );
      })}
    </div>
  );
}
