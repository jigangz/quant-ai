import { useState } from "react";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";
import { Label } from "../../components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../../components/ui/select";
import ErrorState from "../../components/ErrorState";
import { usePlaceOrder } from "../../api/queries";

export default function OrderForm() {
  const [ticker, setTicker] = useState("AAPL");
  const [side, setSide] = useState("buy");
  const [type, setType] = useState("market");
  const [qty, setQty] = useState(10);
  const [price, setPrice] = useState("");
  const place = usePlaceOrder();

  const submit = async (e) => {
    e.preventDefault();
    const payload = { ticker, side, order_type: type, quantity: Number(qty) };
    if (type === "limit") payload.limit_price = Number(price);
    await place.mutateAsync(payload);
  };

  return (
    <form onSubmit={submit} className="space-y-3">
      <div>
        <Label htmlFor="ticker">Ticker</Label>
        <Input id="ticker" value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} />
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <Label>Side</Label>
          <Select value={side} onValueChange={setSide}>
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="buy">Buy</SelectItem>
              <SelectItem value="sell">Sell</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label>Type</Label>
          <Select value={type} onValueChange={setType}>
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="market">Market</SelectItem>
              <SelectItem value="limit">Limit</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <Label htmlFor="qty">Quantity</Label>
          <Input id="qty" type="number" min="1" value={qty} onChange={(e) => setQty(e.target.value)} />
        </div>
        {type === "limit" && (
          <div>
            <Label htmlFor="price">Limit Price</Label>
            <Input id="price" type="number" step="0.01" value={price} onChange={(e) => setPrice(e.target.value)} />
          </div>
        )}
      </div>
      <Button type="submit" disabled={place.isPending} className="w-full">
        {place.isPending ? "Placing..." : `Place ${side.toUpperCase()}`}
      </Button>
      {place.error && <ErrorState error={place.error} />}
      {place.data && <div className="text-sm text-up">Order placed · id: {place.data.order_id || place.data.id}</div>}
    </form>
  );
}
