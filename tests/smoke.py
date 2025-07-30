from mmstory import expand_prompt, scene_json_from_outline, render_scene_prose, generate_dialogue, style_transfer

seed = "A shy linguistics student discovers a dead language can summon storms."
brief = expand_prompt(seed)
print("Brief Generated")
print(len(brief))

print("Generating Prose...")
sc = scene_json_from_outline(brief, beat_index=1)
prose = render_scene_prose(sc, target_len=180)

print("Prose Generated")
print(len(prose))

print("Generating Dialog...")
chars = [
    {"name":"Mira","role":"protagonist","objective":"test the storm-chant safely","emotion":"guarded"},
    {"name":"Arjun","role":"mentor","objective":"discourage reckless use","emotion":"anxious"},
]
dlg = generate_dialogue(chars, "Negotiate boundaries for trying the chant on the pier.", "curiosity vs caution", turns=8)
print("Dialog Generated")

print("Transfering the Style...")
styled = style_transfer(prose, "magical realism with lightly lyrical cadence, present tense")
print("Style Transferred")

print(len(brief), len(prose), len(dlg), len(styled))
print("==BRIEF==")
print(brief)
print("==PROSE==")
print(prose)
print("==DIALOG==")
print(dlg)
print("==STYLED==")
print(styled)