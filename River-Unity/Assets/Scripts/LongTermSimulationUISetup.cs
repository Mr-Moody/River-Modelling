using UnityEngine;
using UnityEngine.UI;
using TMPro;

#if UNITY_EDITOR
using UnityEditor;
#endif

/// <summary>
/// Helper script to automatically create a Long-Term Simulation UI structure.
/// Attach this to any GameObject and click "Create UI" in the Inspector to generate the UI.
/// </summary>
public class LongTermSimulationUISetup : MonoBehaviour
{
#if UNITY_EDITOR
    [ContextMenu("Create Long-Term Simulation UI")]
    public void CreateUI()
    {
        // Create Canvas
        GameObject canvasObj = new GameObject("LongTermSimulationCanvas");
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
        mainRect.sizeDelta = new Vector2(450, 250);
        mainRect.anchoredPosition = new Vector2(0, 200); // Position at top center
        
        // Add layout
        VerticalLayoutGroup layout = mainPanel.AddComponent<VerticalLayoutGroup>();
        layout.spacing = 15;
        layout.padding = new RectOffset(20, 20, 20, 20);
        layout.childControlHeight = false;
        layout.childControlWidth = true;
        layout.childForceExpandWidth = true;
        
        // Create title
        CreateLabel(mainPanel.transform, "Long-Term Simulation (30 Years)", 20);
        
        // Create time step selection section
        GameObject timeStepSection = new GameObject("TimeStepSection");
        timeStepSection.transform.SetParent(mainPanel.transform, false);
        
        VerticalLayoutGroup timeStepLayout = timeStepSection.AddComponent<VerticalLayoutGroup>();
        timeStepLayout.spacing = 5;
        timeStepLayout.childControlHeight = false;
        timeStepLayout.childControlWidth = true;
        timeStepLayout.childForceExpandWidth = true;
        
        CreateLabel(timeStepSection.transform, "Time Step:", 14);
        CreateDropdown(timeStepSection.transform, "TimeStepDropdown");
        
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
        
        // Create progress text
        CreateText(progressSection.transform, "ProgressText", "Ready to start", 16);
        
        // Add LongTermSimulationUI component
        LongTermSimulationUI ui = mainPanel.AddComponent<LongTermSimulationUI>();
        
        Debug.Log("[LongTermSimulationUISetup] UI structure created!");
        Debug.Log("[LongTermSimulationUISetup] Please assign the LongTermSimulationController reference in the LongTermSimulationUI component.");
        
        Selection.activeGameObject = mainPanel;
    }
    
    private GameObject CreatePanel(Transform parent, string name, Color color)
    {
        GameObject panel = new GameObject(name);
        panel.transform.SetParent(parent, false);
        
        Image img = panel.AddComponent<Image>();
        img.color = color;
        
        RectTransform rect = panel.GetComponent<RectTransform>();
        rect.sizeDelta = new Vector2(450, 250);
        
        return panel;
    }
    
    private void CreateLabel(Transform parent, string text, int fontSize)
    {
        GameObject label = new GameObject("Label_" + text.Replace(" ", "").Replace("(", "").Replace(")", ""));
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
    
    private void CreateDropdown(Transform parent, string name)
    {
        GameObject dropdownObj = new GameObject(name);
        dropdownObj.transform.SetParent(parent, false);
        
        Dropdown dropdown = dropdownObj.AddComponent<Dropdown>();
        
        // Background
        GameObject bg = new GameObject("Background");
        bg.transform.SetParent(dropdownObj.transform, false);
        Image bgImg = bg.AddComponent<Image>();
        bgImg.color = new Color(0.3f, 0.3f, 0.3f, 1f);
        RectTransform bgRect = bg.GetComponent<RectTransform>();
        bgRect.anchorMin = Vector2.zero;
        bgRect.anchorMax = Vector2.one;
        bgRect.sizeDelta = Vector2.zero;
        dropdown.targetGraphic = bgImg;
        
        // Label (Dropdown requires UnityEngine.UI.Text, not TextMeshProUGUI)
        GameObject label = new GameObject("Label");
        label.transform.SetParent(dropdownObj.transform, false);
        Text labelText = label.AddComponent<Text>();
        labelText.text = "1 day";
        labelText.fontSize = 14;
        labelText.color = Color.white;
        labelText.alignment = TextAnchor.MiddleLeft;
        RectTransform labelRect = label.GetComponent<RectTransform>();
        labelRect.anchorMin = Vector2.zero;
        labelRect.anchorMax = Vector2.one;
        labelRect.offsetMin = new Vector2(10, 2);
        labelRect.offsetMax = new Vector2(-25, -2);
        dropdown.captionText = labelText;
        
        // Arrow
        GameObject arrow = new GameObject("Arrow");
        arrow.transform.SetParent(dropdownObj.transform, false);
        Image arrowImg = arrow.AddComponent<Image>();
        arrowImg.color = Color.white;
        RectTransform arrowRect = arrow.GetComponent<RectTransform>();
        arrowRect.anchorMin = new Vector2(1f, 0.5f);
        arrowRect.anchorMax = new Vector2(1f, 0.5f);
        arrowRect.pivot = new Vector2(1f, 0.5f);
        arrowRect.sizeDelta = new Vector2(20, 20);
        arrowRect.anchoredPosition = new Vector2(-10, 0);
        dropdown.captionImage = arrowImg;
        
        // Create dropdown template (required for dropdown to work)
        GameObject template = new GameObject("Template");
        template.transform.SetParent(dropdownObj.transform, false);
        template.SetActive(false); // Template should be inactive by default
        
        Image templateBg = template.AddComponent<Image>();
        templateBg.color = new Color(0.2f, 0.2f, 0.2f, 1f);
        RectTransform templateRect = template.GetComponent<RectTransform>();
        templateRect.anchorMin = new Vector2(0f, 0f);
        templateRect.anchorMax = new Vector2(1f, 0f);
        templateRect.pivot = new Vector2(0.5f, 1f);
        templateRect.anchoredPosition = Vector2.zero;
        templateRect.sizeDelta = new Vector2(0, 150); // Height for dropdown list
        
        // Add ScrollRect for scrolling if many items
        ScrollRect scrollRect = template.AddComponent<ScrollRect>();
        scrollRect.horizontal = false;
        scrollRect.vertical = true;
        
        // Create viewport
        GameObject viewport = new GameObject("Viewport");
        viewport.transform.SetParent(template.transform, false);
        Image viewportMask = viewport.AddComponent<Image>();
        Mask mask = viewport.AddComponent<Mask>();
        mask.showMaskGraphic = false;
        RectTransform viewportRect = viewport.GetComponent<RectTransform>();
        viewportRect.anchorMin = Vector2.zero;
        viewportRect.anchorMax = Vector2.one;
        viewportRect.sizeDelta = Vector2.zero;
        viewportRect.anchoredPosition = Vector2.zero;
        scrollRect.viewport = viewportRect;
        
        // Create content area
        GameObject content = new GameObject("Content");
        content.transform.SetParent(viewport.transform, false);
        RectTransform contentRect = content.AddComponent<RectTransform>(); // Explicitly add RectTransform
        contentRect.anchorMin = new Vector2(0f, 1f);
        contentRect.anchorMax = new Vector2(1f, 1f);
        contentRect.pivot = new Vector2(0.5f, 1f);
        contentRect.sizeDelta = new Vector2(0, 28); // Height per item
        contentRect.anchoredPosition = Vector2.zero;
        scrollRect.content = contentRect;
        
        // Add VerticalLayoutGroup to content
        VerticalLayoutGroup contentLayout = content.AddComponent<VerticalLayoutGroup>();
        contentLayout.childControlHeight = true;
        contentLayout.childControlWidth = true;
        contentLayout.childForceExpandHeight = false;
        contentLayout.childForceExpandWidth = true;
        contentLayout.spacing = 0;
        
        // Create item template (Toggle)
        GameObject item = new GameObject("Item");
        item.transform.SetParent(content.transform, false);
        RectTransform itemRect = item.AddComponent<RectTransform>(); // Add RectTransform first
        itemRect.sizeDelta = new Vector2(0, 28);
        
        Toggle itemToggle = item.AddComponent<Toggle>();
        itemToggle.isOn = false;
        
        // Item background
        GameObject itemBg = new GameObject("Item Background");
        itemBg.transform.SetParent(item.transform, false);
        Image itemBgImg = itemBg.AddComponent<Image>();
        itemBgImg.color = new Color(0.3f, 0.3f, 0.3f, 1f);
        RectTransform itemBgRect = itemBg.GetComponent<RectTransform>();
        itemBgRect.anchorMin = Vector2.zero;
        itemBgRect.anchorMax = Vector2.one;
        itemBgRect.sizeDelta = Vector2.zero;
        itemToggle.targetGraphic = itemBgImg;
        
        // Item label
        GameObject itemLabel = new GameObject("Item Label");
        itemLabel.transform.SetParent(item.transform, false);
        Text itemLabelText = itemLabel.AddComponent<Text>();
        itemLabelText.text = "Option";
        itemLabelText.fontSize = 14;
        itemLabelText.color = Color.white;
        itemLabelText.alignment = TextAnchor.MiddleLeft;
        RectTransform itemLabelRect = itemLabel.GetComponent<RectTransform>();
        itemLabelRect.anchorMin = Vector2.zero;
        itemLabelRect.anchorMax = Vector2.one;
        itemLabelRect.offsetMin = new Vector2(10, 0);
        itemLabelRect.offsetMax = new Vector2(-10, 0);
        
        // Item checkmark (optional, but good to have)
        GameObject itemCheckmark = new GameObject("Item Checkmark");
        itemCheckmark.transform.SetParent(item.transform, false);
        Image itemCheckmarkImg = itemCheckmark.AddComponent<Image>();
        itemCheckmarkImg.color = new Color(0.2f, 0.8f, 0.2f, 1f);
        RectTransform itemCheckmarkRect = itemCheckmark.GetComponent<RectTransform>();
        itemCheckmarkRect.anchorMin = new Vector2(1f, 0.5f);
        itemCheckmarkRect.anchorMax = new Vector2(1f, 0.5f);
        itemCheckmarkRect.pivot = new Vector2(1f, 0.5f);
        itemCheckmarkRect.sizeDelta = new Vector2(20, 20);
        itemCheckmarkRect.anchoredPosition = new Vector2(-10, 0);
        itemToggle.graphic = itemCheckmarkImg;
        
        // Assign template to dropdown
        dropdown.template = templateRect;
        
        RectTransform dropdownRect = dropdownObj.GetComponent<RectTransform>();
        dropdownRect.sizeDelta = new Vector2(0, 30);
        
        // Add options
        dropdown.options.Clear();
        for (int i = 1; i <= 7; i++)
        {
            dropdown.options.Add(new Dropdown.OptionData($"{i} day{(i > 1 ? "s" : "")}"));
        }
        dropdown.value = 0;
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
