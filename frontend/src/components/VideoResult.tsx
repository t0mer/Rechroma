/** Result player for a colorized video job. */
export function VideoResult({ src }: { src: string }) {
  return (
    <video
      src={src}
      controls
      playsInline
      className="aspect-video w-full rounded-lg border border-border bg-black"
    />
  );
}
