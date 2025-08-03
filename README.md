# gemma3n-experiments

© 2025 [Ryan Mehra]. All rights reserved.

This code is proprietary and may not be copied, modified, distributed, or used in any way without express written permission from the author.

____

modality/text/archery_vision_analysis.py: Provide image files in the format: png, jpg, jpeg, webp –– in the same folder as code file, we can't distribute image files to avoid copyright issue. However, you can use your own image or any publicaly avaible images from the web.

modality/vision/LLMManager.swift: 
- This library can be called with appropriate archer's completition data, examples are shared in 'examples' folder.
- Sample inference code:

  > @StateObject private var llmManager = LLMManager()
  > llmManager.generateAnalysis(systemPrompt: "", userCompetitionSummaryData: competitionJSON)
