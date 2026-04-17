import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "../../components/ui/accordion";
import { Badge } from "../../components/ui/badge";

export default function SimilarCasesList({ results }) {
  if (!results || results.length === 0) {
    return <p className="text-sm text-muted">No similar cases found.</p>;
  }
  return (
    <Accordion type="multiple">
      {results.map((r, idx) => (
        <AccordionItem key={idx} value={`case-${idx}`}>
          <AccordionTrigger className="hover:no-underline">
            <div className="flex items-center gap-3">
              <Badge variant="info">{r.score?.toFixed(3) || "—"}</Badge>
              <span className="text-sm text-foreground text-left">
                {r.text?.substring(0, 80)}{r.text?.length > 80 ? "..." : ""}
              </span>
            </div>
          </AccordionTrigger>
          <AccordionContent>
            <p className="text-sm text-muted whitespace-pre-wrap">{r.text}</p>
          </AccordionContent>
        </AccordionItem>
      ))}
    </Accordion>
  );
}
