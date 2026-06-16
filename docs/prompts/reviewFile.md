# Goal

Use the **Graphify** skill to assist with the following workflow.
[<file>] = MANDATORY from argument if not present FAIL sking for it.

## Workflow: Start

```plaintext
/ultracode
```

**Repeat:**

- Caveman: Review the file at
  [<file>]
- Then fix all identified issues.

**Until:** No issues remain.

---

## Workflow: End

```plaintext
/ultracode
```

**Repeat:**

- Caveman: Review the file at
  [<file>]
  using an **adversarial attitude modality**.
- Then fix all identified issues.

**Until:** No issues remain.

---

## Exit Condition

- **Success:** Exit the goal when both `workflow_start` and `workflow_end` complete successfully.
- **Failure:** Otherwise, restart from `workflow_start`.
