using UnityEngine;
using UnityEngine.UI;
using TMPro;

#if UNITY_EDITOR
using UnityEditor;
#endif

/// <summary>
/// Helper script to automatically create a simplified Monte Carlo UI structure.
/// Attach this to any GameObject and click "Create UI" in the Inspector to generate the UI.
/// </summary>
public class MonteCarloUISetup : MonoBehaviour
{
#if UNITY_EDITOR
    [ContextMenu("Create Monte Carlo UI")]
    public void CreateUI()
    {
        // Create Canvas
        GameObject canvasObj = new GameObject("MonteCarloCanvas");
        Canvas canvas = canvasObj.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvasObj.AddComponent<CanvasScaler>();
        canvasObj.AddComponent<GraphicRaycaster>();
        
        // Create main panel
        GameObject mainPanel = CreatePanel(canvasObj.transform, "MainPanel", new Color(0.2f, 0.2f, 0.2f, 0.9f));
        RectTransform mainRect = mainPanel.GetComponent<RectTransform>();
        mainRect.anchorMin = new Vector2(0.5f, 0.5f);
        mainRect.anchorMax = new Vector2(0.5f, 0.5f);
        mainRect.pivot = new Vector2(0.5f, 0.5f);
        mainRect.sizeDelta = new Vector2(400, 200);
        mainRect.anchoredPosition = new Vector2(0, 200); // Position at top center
        
        // Add layout
        VerticalLayoutGroup layout = mainPanel.AddComponent<VerticalLayoutGroup>();
        layout.spacing = 15;
        layout.padding = new RectOffset(20, 20, 20, 20);
        layout.childControlHeight = false;
        layout.childControlWidth = true;
        layout.childForceExpandWidth = true;
        
        // Create title
        CreateLabel(mainPanel.transform, "Monte Carlo Simulation", 20);
        
        // Create button container
        GameObject buttonContainer = new GameObject("ButtonContainer");
        buttonContainer.transform.SetParent(mainPanel.transform, false);
        
        HorizontalLayoutGroup buttonLayout = buttonContainer.AddComponent<HorizontalLayoutGroup>();
        buttonLayout.spacing = 10;
        buttonLayout.childControlHeight = true;
        buttonLayout.childControlWidth = true;
        buttonLayout.childForceExpandWidth = true;
        buttonLayout.childForceExpandHeight = true;
        
        RectTransform buttonContainerRect = buttonContainer.GetComponent<RectTransform>();
        buttonContainerRect.sizeDelta = new Vector2(0, 50);
        
        // Create Start button
        GameObject startBtn = CreateButton(buttonContainer.transform, "StartButton", "Start", new Color(0.2f, 0.6f, 0.2f));
        
        // Create Stop button
        GameObject stopBtn = CreateButton(buttonContainer.transform, "StopButton", "Stop", new Color(0.6f, 0.2f, 0.2f));
        
        // Create Render Toggle button
        GameObject renderBtn = CreateButton(buttonContainer.transform, "RenderToggleButton", "Disable Rendering", new Color(0.4f, 0.4f, 0.4f));
        
        // Create progress section
        GameObject progressSection = new GameObject("ProgressSection");
        progressSection.transform.SetParent(mainPanel.transform, false);
        
        VerticalLayoutGroup progressLayout = progressSection.AddComponent<VerticalLayoutGroup>();
        progressLayout.spacing = 5;
        progressLayout.childControlHeight = false;
        progressLayout.childControlWidth = true;
        progressLayout.childForceExpandWidth = true;
        
        // Create progress bar
        CreateProgressBar(progressSection.transform);
        
        // Create stage text
        CreateText(progressSection.transform, "StageText", "Stage: 0 / 0", 16);
        
        // Add MonteCarloUI component
        MonteCarloUI ui = mainPanel.AddComponent<MonteCarloUI>();
        
        Debug.Log("[MonteCarloUISetup] Simplified UI structure created!");
        Debug.Log("[MonteCarloUISetup] Please assign the Monte Carlo Manager reference in the MonteCarloUI component.");
        
        Selection.activeGameObject = mainPanel;
    }
    
    private GameObject CreatePanel(Transform parent, string name, Color color)
    {
        GameObject panel = new GameObject(name);
        panel.transform.SetParent(parent, false);
        
        Image img = panel.AddComponent<Image>();
        img.color = color;
        
        RectTransform rect = panel.GetComponent<RectTransform>();
        rect.sizeDelta = new Vector2(400, 200);
        
        return panel;
    }
    
    private void CreateLabel(Transform parent, string text, int fontSize)
    {
        GameObject label = new GameObject("Label_" + text.Replace(" ", ""));
        label.transform.SetParent(parent, false);
        
        TextMeshProUGUI tmp = label.AddComponent<TextMeshProUGUI>();
        tmp.text = text;
        tmp.fontSize = fontSize;
        tmp.color = Color.white;
        tmp.fontStyle = FontStyles.Bold;
        tmp.alignment = TextAlignmentOptions.Center;
        
        RectTransform rect = label.GetComponent<RectTransform>();
        rect.sizeDelta = new Vector2(0, fontSize + 10);
    }
    
    private GameObject CreateButton(Transform parent, string name, string text, Color color)
    {
        GameObject btn = new GameObject(name);
        btn.transform.SetParent(parent, false);
        
        Image img = btn.AddComponent<Image>();
        img.color = color;
        
        Button button = btn.AddComponent<Button>();
        ColorBlock colors = button.colors;
        colors.normalColor = color;
        colors.highlightedColor = color * 1.2f;
        colors.pressedColor = color * 0.8f;
        button.colors = colors;
        
        RectTransform rect = btn.GetComponent<RectTransform>();
        rect.sizeDelta = new Vector2(0, 40);
        
        GameObject textObj = new GameObject("Text");
        textObj.transform.SetParent(btn.transform, false);
        TextMeshProUGUI tmp = textObj.AddComponent<TextMeshProUGUI>();
        tmp.text = text;
        tmp.fontSize = 16;
        tmp.color = Color.white;
        tmp.alignment = TextAlignmentOptions.Center;
        
        RectTransform textRect = textObj.GetComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.sizeDelta = Vector2.zero;
        
        return btn;
    }
    
    private void CreateProgressBar(Transform parent)
    {
        GameObject sliderObj = new GameObject("ProgressBar");
        sliderObj.transform.SetParent(parent, false);
        
        Slider slider = sliderObj.AddComponent<Slider>();
        slider.minValue = 0f;
        slider.maxValue = 100f;
        slider.value = 0f;
        
        RectTransform sliderRect = sliderObj.GetComponent<RectTransform>();
        sliderRect.sizeDelta = new Vector2(0, 25);
        
        // Background
        GameObject bg = new GameObject("Background");
        bg.transform.SetParent(sliderObj.transform, false);
        Image bgImg = bg.AddComponent<Image>();
        bgImg.color = new Color(0.2f, 0.2f, 0.2f, 1f);
        RectTransform bgRect = bg.GetComponent<RectTransform>();
        bgRect.anchorMin = Vector2.zero;
        bgRect.anchorMax = Vector2.one;
        bgRect.sizeDelta = Vector2.zero;
        slider.targetGraphic = bgImg;
        
        // Fill
        GameObject fill = new GameObject("Fill Area");
        fill.transform.SetParent(sliderObj.transform, false);
        RectTransform fillAreaRect = fill.AddComponent<RectTransform>();
        fillAreaRect.anchorMin = Vector2.zero;
        fillAreaRect.anchorMax = Vector2.one;
        fillAreaRect.sizeDelta = Vector2.zero;
        fillAreaRect.offsetMin = new Vector2(2, 2);
        fillAreaRect.offsetMax = new Vector2(-2, -2);
        
        GameObject fillImg = new GameObject("Fill");
        fillImg.transform.SetParent(fill.transform, false);
        Image fillImage = fillImg.AddComponent<Image>();
        fillImage.color = new Color(0.2f, 0.5f, 0.8f, 1f);
        RectTransform fillRect = fillImg.GetComponent<RectTransform>();
        fillRect.anchorMin = Vector2.zero;
        fillRect.anchorMax = new Vector2(0f, 1f);
        fillRect.sizeDelta = Vector2.zero;
        
        slider.fillRect = fillRect;
        
        // No handle needed for progress bar
    }
    
    private void CreateText(Transform parent, string name, string text, int fontSize)
    {
        GameObject textObj = new GameObject(name);
        textObj.transform.SetParent(parent, false);
        
        TextMeshProUGUI tmp = textObj.AddComponent<TextMeshProUGUI>();
        tmp.text = text;
        tmp.fontSize = fontSize;
        tmp.color = Color.white;
        tmp.alignment = TextAlignmentOptions.Center;
        
        RectTransform rect = textObj.GetComponent<RectTransform>();
        rect.sizeDelta = new Vector2(0, fontSize + 5);
    }
#endif
}
