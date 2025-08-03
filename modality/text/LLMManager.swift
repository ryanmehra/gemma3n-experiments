/**
 * Ownership: Ryan Mehra @ ArcheryPulse
 * Created: 2025-07-14
 *
 * License: Proprietary and Confidential
 *
 * This code defines a manager for handling interactions with a large language model (LLM) using the MediaPipeTasksGenAI framework.
 * It includes model loading, inference, and response handling.
 *
 * This file contains confidential and proprietary information
 * of ArcheryPulse. Unauthorized copying, modification,
 * distribution, or disclosure of this file or its contents
 * is strictly prohibited without prior written consent.
 */


import Foundation
import MediaPipeTasksGenAI

enum ModelType: String, CaseIterable {
    case onDevice = "On-Device"
    case api = "API-based"
}

@MainActor
class LLMManager: ObservableObject {
    @Published var llmResponse: String = ""
    @Published var error: Error?
    
    private var llmInference: LlmInference?
    
    init() {
        // No-op: model will be loaded on demand to defer loading cost
    }

    /// Ensures the LLM model is loaded before inference.
    private func ensureModelLoaded() {
        guard llmInference == nil else { return }
        let loadStart = Date()
        // Locate the model file in the app bundle
        guard let bundleModelURL = Bundle.main.url(forResource: "gemma-3n-E4B-it-int4", withExtension: "task") else {
            let err = NSError(domain: "LLMManager", code: -1,
                              userInfo: [NSLocalizedDescriptionKey: "Model file gemma-3n-E4B-it-int4.task not found"])
            self.error = err
            print("LLMManager loadModel error: \(err.localizedDescription)")
            return
        }
        // Copy model to Application Support directory if needed (to allow XNNPACK cache file creation)
        let fileManager = FileManager.default
        let appSupportDir = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        // Ensure Application Support directory exists
        if !fileManager.fileExists(atPath: appSupportDir.path) {
            do { try fileManager.createDirectory(at: appSupportDir, withIntermediateDirectories: true, attributes: nil) } catch { print("Failed to create Application Support directory: \(error)") }
        }
        let destURL = appSupportDir.appendingPathComponent("gemma-3n-E4B-it-int4.task")
        if !fileManager.fileExists(atPath: destURL.path) {
            do {
                try fileManager.copyItem(at: bundleModelURL, to: destURL)
                print("Copied model to Application Support: \(destURL.path)")
            } catch {
                // If file exists, remove and retry copy (handle app updates)
                if (error as NSError).code == NSFileWriteFileExistsError {
                    do {
                        try fileManager.removeItem(at: destURL)
                        try fileManager.copyItem(at: bundleModelURL, to: destURL)
                        print("Re-copied model to Application Support: \(destURL.path)")
                    } catch {
                        self.error = error
                        print("Failed to copy model to Application Support: \(error)")
                        return
                    }
                } else {
                    self.error = error
                    print("Failed to copy model to Application Support: \(error)")
                    return
                }
            }
        }
        do {
            var opts = MediaPipeTasksGenAI.LlmInference.Options(modelPath: destURL.path)
            opts.maxTokens = 5000
            llmInference = try LlmInference(options: opts)
            let loadElapsed = Date().timeIntervalSince(loadStart)
            print("\n\nGemma3n loaded model at", destURL.path)
            print(String(format: "\n\nGemma3n model load time: %.2f seconds", loadElapsed))
        } catch {
            self.error = error
            print("Failed to load LlmInference:", error)
        }
    }

    /// Calls the LLM with system/user sections using a session and custom decoding options.
    @MainActor
    func generateAnalysis(systemPrompt: String, userCompetitionSummaryData: String) async {
        // Load model on demand
        ensureModelLoaded()
        guard let llm = llmInference else {
            self.error = NSError(domain: "LLMManager", code: -1,
                                 userInfo: [NSLocalizedDescriptionKey: "LlmInference not initialized."])
            return
        }

let systemPrompt = """
You are **ArcheryAI**, a concise mobile‐style assistant coach.  
You will be given a single JSON object that strictly matches this schema (no extra keys, no missing fields):

{
  "tournament": string,
  "category": string,
  "archer": string,
  "totalScore": number,
  "arrowAverage": number,
  "breakdown": {
    "Xs": number,
    "tens": number,
    "nines": number
  },
  "rounds": [
    {
      "roundNumber": number,
      "ends": [
        {
          "end": number,
          "total": number,
          "arrows": [number, …]
        }
      ]
    }
  ]
}

Your job is to analyze that data and return ONLY a JSON object with exactly these four keys:
  1.	whats_working – an array of up to 2 objects with keys:
    •	round (number)
    •	ends (string, e.g. “1–3”)
    •	category (one of: “Release, “Timing, “Practice”, “Equipment”)
    •	description (string)
  2.	focus_areas – an array of 2–3 objects with keys:
    •	round (number)
    •	ends (string)
    •	category (one of: “Release, “Timing”, “Practice”, “Equipment”, “Physical”, “Environmental”)
    •	issue (string)
  3.	coach_recommends – an array of up to 3 objects with keys:
    •	drill (string, one of the provided drills)
    •	purpose (string)
  4.	pro_tip – a single string of ≤15 words (no surrounding quotes).

Analysis Logic
  •	whats_working: pick the top 2 ends by total across all rounds.

    Allowed Categories:  
       - Release (e.g., stable anchor → consistent form)  
       - Timing (e.g., steady pacing → smooth rhythm)  
       - Practice (e.g., effective warm-up → consistent early scores)  
       - Equipment (e.g., well-tuned bow → no low scores)

  •	focus_areas: pick the 2 ends with the largest drop below arrowAverage.

      Mapped Issues:  
       - Release: Anchor-point discipline, Follow-through maintenance  
       - Timing: Shot timing inconsistency, Release rhythm  
       - Practice: Mental stamina dips (mid-round), End-transition recovery, Warm-up routine, Focus & visualization  
       - Equipment: Bow tuning, Sight calibration, Arrow weight balance, String tension  
       - Physical: Core stability, Upper-body endurance  
       - Environmental: Wind adjustment, Lighting adaptation

  •	coach_recommends: choose drills relevant to the identified categories.

     (Shot-timer drill) – stabilize shot rhythm  
     (Anchor-point drill) – maintain consistent anchor  
     (Bow-tuning check) – inspect limb alignment  
     (7:14 rhythm drill) – build stamina  
     (Wind-adjustment drill) – practice under variable winds  
     (Strength-training drill) – improve core and shoulder stability  
     (Breathing-focus drill) – regulate breathing for calm release  
     (Equipment-check drill) – systematic gear inspection before shooting  

  •	pro_tip: one concise recommendation.

Few-Shot Example

Input:
{
  "tournament": "Sample Shoot",
  "category": "Recurve U18 Men",
  "archer": "Jane Doe",
  "totalScore": 360,
  "arrowAverage": 6.0,
  "breakdown": { "Xs": 1, "tens": 6, "nines": 3 },
  "rounds": [
    {
      "roundNumber": 1,
      "ends": [ { "end": 1, "total": 30, "arrows": [5,5,5,5,5,5] } ]
    }
  ]
}

Output:
{
  "whats_working": [
    {
      "round": 1,
      "ends": "1–1",
      "category": "Practice",
      "description": "Consistent six-arrow grouping demonstrates solid warm-up routine."
    }
  ],
  "focus_areas": [
    {
      "round": 1,
      "ends": "1–1",
      "category": "Timing",
      "issue": "All arrows fired too quickly, causing uneven rhythm."
    }
  ],
  "coach_recommends": [
    {
      "drill": "Shot-timer drill",
      "purpose": "stabilize shot rhythm"
    }
  ],
  "pro_tip": "Focus on a smooth draw and consistent release speed"
}

Hard Rules
  •	Output only the JSON object—no Markdown, no commentary, no code fences.
  •	Do not invent fields or perform calculations beyond the schema.
  •	Omit any round or end with a total of zero.
  •	Use exactly the keys: whats_working, focus_areas, coach_recommends, pro_tip.

"""
        let promptText = """
        <|system|>\(systemPrompt)
        <|user|>\(userCompetitionSummaryData)
        """
        print("\n\n User Competition Data: \(userCompetitionSummaryData)")
        // print("LLMManager analysis FINAL prompt: \(promptText)")
        do {
            let start = Date()
            let response = try await llm.generateResponse(inputText: promptText)
            let elapsed = Date().timeIntervalSince(start)
            self.llmResponse = response
            print("\n\nGemma3n Derived Analysis (JSON): \(response)")
            print(String(format: "\n\nGemma3n analysis inference time: %.2f seconds", elapsed))
        } catch {
            self.error = error
        }
    }
}
