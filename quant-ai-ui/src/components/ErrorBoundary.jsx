import { Component } from "react";
import { Button } from "./ui/button";
import { AlertTriangle } from "lucide-react";

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error("ErrorBoundary caught:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4 p-6 text-center">
          <AlertTriangle className="h-12 w-12 text-down" />
          <div>
            <h2 className="text-xl font-semibold">Something broke</h2>
            <p className="text-sm text-muted mt-1">{String(this.state.error?.message || this.state.error)}</p>
          </div>
          <Button onClick={() => this.setState({ hasError: false, error: null })}>
            Try again
          </Button>
        </div>
      );
    }
    return this.props.children;
  }
}
