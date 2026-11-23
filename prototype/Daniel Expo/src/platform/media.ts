/* Native (iOS / Android) media helpers. The web versions live in media.web.ts. */
import * as VideoThumbnails from "expo-video-thumbnails";

/* A still of a video for the thumbnail grid and the loading frame. Null if it can't be read.
   expo-video-thumbnails is deprecated in favour of expo-video's generateThumbnailsAsync, but
   still ships with SDK 58 and returns a plain file URI that <Image> can show. */
export async function videoThumbnail(uri: string): Promise<string | null> {
  try {
    const { uri: thumb } = await VideoThumbnails.getThumbnailAsync(uri, { time: 100, quality: 0.6 });
    return thumb;
  } catch {
    return null;
  }
}

/* The web build rejects clearly unusable photos (too dark, washed out, blurry) before upload by
   reading pixels on a canvas. React Native has no canvas, so on a phone every photo goes to the
   server, whose mouth gate still rejects unusable ones. */
export async function photoQualityIssue(_uri: string): Promise<string | null> {
  return null;
}
