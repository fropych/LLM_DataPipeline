# Annotation Specification — AG News Text Classification

**Generated:** 2026-03-21 17:15 UTC
**Dataset:** AG News  |  **Rows:** 100
**Task type:** Single-label multi-class text classification
**Classes:** World, Sports, Business, Sci/Tech

---

## 1. Task Description

Given a short news headline + snippet (≤ 400 characters), assign **exactly one** label that best describes the primary topic of the article.

Rules:
- Choose the label that reflects the **dominant theme**, not a minor mention.
- If genuinely ambiguous after reading the full snippet, mark the example as **uncertain**   (flag for human review).
- Do **not** use knowledge outside the provided text — base the decision solely on what is written.

---

## 2. Class Definitions

### World

News about international politics, diplomacy, armed conflicts, governments, heads of state, elections, humanitarian crises, international organisations (UN, NATO, EU), treaties, sanctions, and cross-border events that are primarily geopolitical in nature.

### Sports

News about any competitive sport, including professional leagues, international tournaments, Olympics, individual athletes, team results, transfers, coaching changes, match reports, and sports business directly tied to athletic competition.

### Business

News about companies, financial markets, macroeconomics, trade, corporate strategy (mergers, acquisitions, IPOs), earnings and revenue, stock prices, banking, commodities markets, and economic policy with a primary focus on commerce and finance.

### Sci/Tech

News about scientific research, technology products and companies, software, hardware, the internet, telecommunications, biotechnology, space exploration, medical breakthroughs, patents, and innovation — where the primary angle is technical or scientific.

---

## 3. Examples per Class

### World

**Example 1:**

> White House: Guard Never Disciplined Bush WASHINGTON - President Bush never was disciplined while serving in the Texas Air National Guard, never failed a physical and never asked his father or family friends for help to get him into the Guard during the Vietnam War, the White House said Wednesday....

**Example 2:**

> Families of Iraq War Dead Target Bush in Ads (Reuters) Reuters - Angered by President Bush's policy\in Iraq, a group of military families whose relatives died\there is targeting the president in new television ads to be\aired ahead of the Nov. 2 election.

**Example 3:**

> Some GOP Gay Rights Backers Torn Over Bush (AP) AP - On the eve of the Republican convention, some GOP gay rights supporters frustrated by their party platform's hardline stance against same-sex unions say they are torn over whether to back President Bush in this fall's election.

**Example 4:**

> More talks scheduled as Italy presses Libya on immigration crisis (AFP) AFP - Italian Prime Minister Silvio Berlusconi is aiming to make cooperation with Libya a model for Europe's efforts to stem a tide of illegal immigration from Africa, but further talks are still needed between Tripoli, Rome and...

**Example 5:**

> Bush defends tax cuts WASHINGTON, Oct 2 (AFP) - US President George W. Bush said Saturday he would renew some of the huge tax cuts that form a cornerstone of his economic rejuvenation policy and chided Democratic challenger John Kerry for opposing the cuts.

### Sports

**Example 1:**

> Pantano replaced by Glock Jordan have terminated the contract of Italian driver Giorgio Pantano and called in Timo Glock as a replacement. The team said contractual difficulties were behind the split, and confirmed German Glock would race in Sunday #39;s inaugural Chinese Grand Prix.

**Example 2:**

> Fighting to Save Plan Mayor Anthony Williams, pushing for baseball on the Anacostia, has canceled a meeting Monday with the D.C. Council he must sway.

**Example 3:**

> Carter #39;s  #39;got nothing to say #39; after scoring 15 After hearing rare boos from the home fans, Vince Carter didn #39;t want to talk about it. Carter had 15 points in his first game since demanding to be traded, but the Toronto Raptors lost their

**Example 4:**

> Wagner Suspended After Being Ejected (AP) AP - Philadelphia Phillies closer Billy Wagner was suspended for two games Friday by major league baseball and fined an undisclosed amount after he was ejected against the New York Mets last weekend.

**Example 5:**

> Sportsview: Clemens at Fenway Not to Be (AP) AP - All of New England would have been quaking with anticipation. Just imagine: Roger Clemens back at Fenway, wearing the enemy's colors for a fourth time in the postseason, this time in Game 7 of the World Series.

### Business

**Example 1:**

> Ford Expands Recall of SUVs to 600,000  DETROIT (Reuters) - Ford Motor Co. &lt;A HREF="http://www.investor.reuters.com/FullQuote.aspx?ticker=F.N target=/stocks/quickinfo/fullquote"&gt;F.N&lt;/A&gt; has expanded its  recall of Escape and Mazda Tribute SUVs to nearly 600,000  vehicles, according to th...

**Example 2:**

> Johnson Controls Profit Up 25 Percent Johnson Controls Inc. on Tuesday said its fourth-quarter profit rose 25 percent, helped by a lower tax rate and decreased miscellaneous net expenses.

**Example 3:**

> Advocate Or Promoter? (Forbes.com) Forbes.com - As the second employee of InfoSpace, Jean-Remy Facq saw the value of his stock inflate and then collapse. Now he's suing in federal court in Seattle for a  #36;4.5 million refund of 1999 federal income tax. He's just one of 200 tech workers who have hi...

**Example 4:**

> Before the Bell- Calpine Drops 6 Percent  NEW YORK (Reuters) - Shares of Calpine Corp. &lt;A HREF="http://www.investor.reuters.com/FullQuote.aspx?ticker=CPN.N target=/stocks/quickinfo/fullquote"&gt;CPN.N&lt;/A&gt; fell 6  percent before the bell on Tuesday following reports that one  of its joint ve...

**Example 5:**

> Packers trade McKenzie to Saints Mike McKenzie got his wish today when the Green Bay Packers traded the disgruntled cornerback and a future conditional draft pick to the New Orleans Saints for quarterback JT O #39;Sullivan and a second-round pick in next year #39;s draft.

### Sci/Tech

**Example 1:**

> Blocking out the bad QUESTION: What is the best approach to protecting me from ID theft and related activities? Will a software package work best, or must I go to a hardware firewall?

**Example 2:**

> Apple peels wraps off fastest ever iBook G4 Apple has unveiled what it claims is its fastest ever iBook G4 laptop, a \$1,499 1.33GHz PowerPC G4 processor-based device sporting a 14in screen and a DVD burner drive.

**Example 3:**

> Exit Blog, Enter Legg A high-profile departure and a higher-profile purchase mean nothing ever gets boring at Google.

**Example 4:**

> Intel switches off TV chip IN ANOTHER change of plan, semiconductor giant Intel has cancelled a project to develop a chip for projection TVs, saying it plans to focus its resources in other areas.

**Example 5:**

> Tesco steps up RFID efforts Tesco is rolling out radio barcode technology across its 98 Tesco Extra stores to track high-value items between its internal distribution centres and its outlets.

---

## 4. Edge Cases & Decision Rules

### Tech company earnings

If the story is primarily about a tech company's financial results (stock price, revenue, profit), label as **Business**. If it focuses on a new product or technology breakthrough, label as **Sci/Tech**.

### Sports business / transfers

Transfer fees, stadium financing, or team ownership deals are **Business** only when the financial angle dominates. Player transfers discussed in a sporting context are **Sports**.

### Election violence / war protests

Stories that mix political unrest with military action: if troops or weapons are central, use **World**; if the story is about a domestic protest or vote, still **World**.

### Medical research vs. pharma business

A clinical trial or genome study is **Sci/Tech**. A drug company's patent dispute or quarterly earnings is **Business**.

### Oil prices

Oil price movements driven by OPEC decisions or geopolitics → **World** or **Business**. When the primary angle is market/investor impact, use **Business**.

### Olympic host city politics

Use **World** if the story is about a bid decision by a government or political controversy; use **Sports** if the story is about athletes, results, or competition logistics.

---

## 5. Labeling Workflow

1. Read the full text snippet.
2. Identify the **primary** topic (ignore secondary mentions).
3. Apply the class definition. If still unclear, consult Section 4 edge cases.
4. If confidence < 70 %, add the example to the **uncertain** pile for a second annotator.
5. Record label in Label Studio; do **not** skip any example.

---

## 6. Quality Targets

| Metric | Target |
|--------|--------|
| Inter-annotator agreement (Cohen's κ) | ≥ 0.70 |
| Per-class coverage | ≥ 20 % each |
| Uncertain / flagged rate | ≤ 15 % |
