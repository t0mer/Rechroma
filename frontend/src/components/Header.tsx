import { useState } from "react";
import { KeyRound, Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { getToken, setToken } from "@/lib/api";
import { cn } from "@/lib/utils";

interface HeaderProps {
  theme: "light" | "dark";
  onToggleTheme: () => void;
}

/** The chroma mark: a lens aperture split half grey / half warm — the app's job. */
function Mark() {
  return (
    <span className="relative grid h-10 w-10 place-items-center rounded-xl bg-foreground/5 ring-1 ring-border">
      <span
        className="h-5 w-5 rounded-full"
        style={{
          background:
            "conic-gradient(from 210deg, hsl(var(--muted-foreground)) 0deg 180deg, hsl(var(--primary)) 180deg 360deg)",
        }}
      />
      <span className="absolute h-1.5 w-1.5 rounded-full bg-card" />
    </span>
  );
}

export function Header({ theme, onToggleTheme }: HeaderProps) {
  const [showToken, setShowToken] = useState(false);
  const [token, setLocalToken] = useState(getToken());

  const saveToken = () => {
    setToken(token.trim());
    setShowToken(false);
  };

  return (
    <header className="sticky top-0 z-30 border-b border-border/70 bg-background/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3.5 sm:px-6">
        <div className="flex items-center gap-3">
          <Mark />
          <div className="leading-tight">
            <h1 className="font-display text-xl font-bold tracking-tight">
              Rechroma
            </h1>
            <p className="hidden text-xs text-muted-foreground sm:block">
              Bring old photos back to life
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1.5">
          <div className="relative">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowToken((v) => !v)}
              aria-label="API token"
              className={cn(getToken() && "text-primary")}
            >
              <KeyRound className="h-[1.15rem] w-[1.15rem]" />
            </Button>
            {showToken && (
              <div className="absolute right-0 top-11 z-40 w-72 rounded-lg border border-border bg-card p-3 shadow-lift animate-fade-up">
                <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
                  API token (optional)
                </label>
                <input
                  type="password"
                  value={token}
                  onChange={(e) => setLocalToken(e.target.value)}
                  placeholder="Only if WEB_AUTH_TOKEN is set"
                  className="h-9 w-full rounded-md border border-input bg-background px-2.5 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                />
                <div className="mt-2.5 flex justify-end gap-2">
                  <Button size="sm" variant="ghost" onClick={() => setShowToken(false)}>
                    Cancel
                  </Button>
                  <Button size="sm" onClick={saveToken}>
                    Save
                  </Button>
                </div>
              </div>
            )}
          </div>

          <Button
            variant="ghost"
            size="icon"
            onClick={onToggleTheme}
            aria-label={theme === "dark" ? "Switch to light" : "Switch to dark"}
          >
            {theme === "dark" ? (
              <Sun className="h-[1.15rem] w-[1.15rem]" />
            ) : (
              <Moon className="h-[1.15rem] w-[1.15rem]" />
            )}
          </Button>
        </div>
      </div>
    </header>
  );
}
