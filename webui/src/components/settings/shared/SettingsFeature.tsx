import { useEffect, useRef, useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import { useTranslation } from "react-i18next";
import { ToggleButton } from "@/components/settings/ToggleButton";
import { SettingsGroup, SettingsRow } from "@/components/settings/shared/SettingsControls";
import { Dialog, DialogContent, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Disclosure } from "@/components/ui/disclosure";

export function SettingsAdvancedOptions({ children }: { children: ReactNode }) {
  const { t } = useTranslation();
  return (
    <Disclosure
      summaryClassName="settings-list-inset flex min-h-12 cursor-pointer items-center justify-between gap-4 rounded-xl text-[13px] leading-5 text-muted-foreground settings-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      summary={<>
        {t("settings.runtimeConfig.advancedOptions")}
        <ChevronDown aria-hidden className="h-3.5 w-3.5 shrink-0 transition-transform duration-200 group-data-[state=open]/disclosure:rotate-180 motion-reduce:transition-none" />
      </>}
    >
      <div className="space-y-1">{children}</div>
    </Disclosure>
  );
}

export function SettingsFeature({
  title, enabled, onChange, disabled, initialOpen = false, error, children,
}: {
  title: string;
  /** Omit for an always-on feature: the row then only opens its settings. */
  enabled?: boolean;
  onChange?: (enabled: boolean) => void;
  disabled?: boolean;
  initialOpen?: boolean;
  error?: string;
  children?: ReactNode;
}) {
  const { t } = useTranslation();
  const active = enabled ?? true;
  const [open, setOpen] = useState(initialOpen && active);
  const section = useRef<HTMLElement>(null);
  useEffect(() => {
    if (initialOpen && active) {
      setOpen(true);
      section.current?.scrollIntoView?.({ block: "nearest" });
    }
  }, [initialOpen, active]);
  return (
    <section ref={section} aria-label={title}>
      <Dialog open={Boolean(children) && open} onOpenChange={setOpen}>
      <SettingsGroup>
        <SettingsRow title={children ? (
          <DialogTrigger asChild>
          <button type="button"
            className="flex min-h-9 w-full items-center rounded-lg text-start focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
            {title}
          </button>
          </DialogTrigger>
        ) : title}>
          {children ? <DialogTrigger asChild>
            <button type="button"
              className="mr-3 shrink-0 rounded-lg px-2 py-1 text-[13px] leading-5 text-muted-foreground settings-hover hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
              {t("settings.configure")}
            </button>
          </DialogTrigger> : null}
          {onChange ? <ToggleButton checked={active} disabled={disabled} ariaLabel={title} label={title}
            onChange={(next) => { setOpen(next); onChange(next); }} /> : null}
        </SettingsRow>
      </SettingsGroup>
      {children ? <DialogContent aria-describedby={undefined} className="settings-grid max-h-[85dvh] w-[min(calc(100vw-2rem),40rem)] max-w-none overflow-y-auto p-0 [--settings-surface:var(--background)]">
        <DialogTitle className="sr-only">{title}</DialogTitle>
        <div className="pb-4 pt-8">{children}</div>
      </DialogContent> : null}
      </Dialog>
      {error && !open ? <p role="alert" className="settings-list-inset pt-2 text-[13px] leading-5 text-destructive">{error}</p> : null}
    </section>
  );
}
