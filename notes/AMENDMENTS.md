# The 2026 amendment, and the questions I wrote for it

Regulation (EU) 2026/1744, the "Digital Omnibus on AI", was adopted on 8 July
2026, published in the Official Journal on 24 July 2026 and entered into force
on 27 July 2026 (its Article 4: the third day after publication). It amends
2024/1689 directly. The Publications Office has a consolidated text of
2024/1689 dated the same day, CELEX `02024R1689-20260727`.

I fetched both from Cellar the same way `fetch.py` fetches the original
(`http://publications.europa.eu/resource/celex/<CELEX>` with
`Accept-Language: eng`), read the amending act's Article 1 point by point, and
checked each point I rely on against the consolidated text. Nothing below comes
from a summary of the Omnibus; every change was read in one of those two
documents, and the quoted wording is what the consolidated corpus contains.

The consolidated text has no legal effect (it says so at the top). Only the
Official Journal texts are authentic. For a QA system that is the right
trade: the consolidated text is what a reader actually needs, and the OJ
amending act is where I checked it.

## What the Omnibus changes in 2024/1689

From Article 1 of 2026/1744, by point:

| Point | Provision | Change |
|---|---|---|
| (1) | Art 1(2)(g) | innovation support now names SMCs as well as SMEs |
| (2) | Art 2(2), 2(7) | Section B of Annex I: only Art 6(1), Art 60a and Arts 102 to 112 apply; Arts 57 to 59 in so far as integrated. 2(7) now points to Art 4a |
| (3) | Art 2(13) | new: requirements for Art 6(1) systems may be limited where Annex I Section A law gives equivalent protection |
| (4) | Art 3(14), (14a), (14b) | "safety component" redefined around a safety function; new SME and SMC definitions |
| (5) | Art 4 | replaced: take measures to support AI literacy, no specific level guaranteed |
| (6) | Art 4a | new: special categories of personal data for bias detection and correction |
| (7) | Art 5(1)(ba), (bb), 5(1a), 5(1b) | new prohibitions: non-consensual intimate material of identifiable people, and child sexual abuse material |
| (8) | Art 6(1a) to (1c) | what is and is not a safety component |
| (9) | Art 10(1), (5), (6) | 10(5) deleted, its basis moved to Art 4a |
| (10) | Art 11(1) | SMCs may also use the simplified technical documentation |
| (11), (12) | Art 17(2), Art 25(2), 25(4) | proportionality for SMCs; value chain cooperation rewritten |
| (13) to (19) | Arts 27 to 30, 40, 42, 43 | FRIA cross-references, notified body single application, CRA presumption, conformity assessment for Annex I products |
| (20), (21) | Art 50(7), Art 56(6) | codes of practice on marking and labelling assessed by the Commission |
| (22) to (25) | Arts 57, 58, 60, 60a | Union level sandbox run by the AI Office; new Art 60a, real-world testing for Section B products |
| (26) to (34) | Arts 63, 64, 69, 70, 72, 75, 75a to 75d, 76, 77 | AI Office exclusive competence and enforcement powers, new Arts 75a to 75d |
| (35) to (38) | Arts 95, 96, 97, 99 | Art 99(4)(da) new, Art 99(6a): SMC fines capped at whichever is lower |
| (39) | Art 111(2), 111(4) | new: Art 50(2) marking for systems already on the market by 2 December 2026 |
| (40) | Art 113 | new dates, see below |
| (41) to (43) | Annex I, VIII, new Annex XIV | Annex I Section A point 1 deleted; notified body codes in Annex XIV |

The new dates in Article 113, third paragraph, as amended:

- (a) Chapters I and II from 2 February 2025, except Art 5(1)(ba), (bb) and
  Art 5(1a), (1b), which apply from 2 December 2026;
- (c)(i) Chapter III, Sections 1 to 3 (except Art 6(5)) from 2 December 2027
  for Art 6(2) and Annex III systems;
- (c)(ii) the same from 2 August 2028 for Art 6(1) and Annex I systems;
- (d) Articles 102 to 110 from 27 July 2026.

The general date in the second paragraph, 2 August 2026, did not change.

## The 11 new questions

They are in [`eval/qa_amended.jsonl`](../eval/qa_amended.jsonl). Each has gold
units, a reference answer, the amending point in `amended_by`, and the exact
wording its answer rests on in `evidence`. A test checks that every quote is
inside a gold unit of the 2026 corpus and in no chunk of the 2024 corpus, so
none of these can be answered correctly from the original text.

| id | gold | quoted provision |
|---|---|---|
| a01 | art_113 | "2 December 2027 as regards AI systems classified as high-risk pursuant to Article 6(2) and Annex III" |
| a02 | art_113 | "2 August 2028 as regards AI systems classified as high-risk pursuant to Article 6(1) and Annex I" |
| a03 | art_5 | Art 5(1)(ba): "generates or manipulates realistic images, videos, audio or similar material of an identifiable natural person's intimate parts" |
| a04 | art_5, art_113 | the (ba) wording above, and Art 113(a): "points (ba) and (bb), and Article 5(1a) and (1b) which shall apply from 2 December 2026" |
| a05 | art_4 | "This obligation does not require providers or deployers to guarantee any specific level of AI literacy of any individual." |
| a06 | art_4a | Art 4a(2): "Providers and deployers of other AI systems and models and deployers of high-risk AI systems may exceptionally process special categories of personal data" |
| a07 | art_60a | Art 60a(1): "Member States may allow, in accordance with this Article, the testing of high-risk AI systems in real world conditions outside AI regulatory sandboxes by providers or prospective providers of AI enabled products covered by the Union harmonisation legislation listed in Section B of Annex I" |
| a08 | art_6 | Art 6(1a): "solely used for non-safety related aspects of user assistance, performance optimisation, service efficiency, automation or convenience or quality control shall not qualify as safety components" |
| a09 | art_111 | Art 111(4): "that have been placed on the market before 2 August 2026 shall take the necessary steps in order to comply with Article 50(2) by 2 December 2026" |
| a10 | art_99 | Art 99(6a): "In the case of SMCs, each fine referred to in paragraphs 4 and 5 shall be up to the percentages or amount referred therein, whichever is lower." |
| a11 | art_113, art_11 | the a01 wording, and Art 11(1): "SMEs, including start-ups, and SMCs, may provide the elements of the technical documentation specified in Annex IV in a simplified manner" |

I left out one I had drafted, on who supervises an AI system built on a
general-purpose model by the same provider. The amended Article 75(1) makes the
AI Office exclusively competent, but the original already gave the AI Office
that supervision, so the old text answers it nearly the same way and it would
not test anything.

## Original questions the amendment makes stale

Three of the 45 original questions carry a `stale` field now. I kept them as
they were, since they are still correct for the 2024 text they were written
against and the published numbers were measured on that text.

- **m04, superseded.** Article 2(2) was replaced. For Section B products it is
  now Art 6(1), Art 60a and Arts 102 to 112, with Arts 57, 58 and 59 in so far
  as integrated, not "Article 6(1), Articles 102 to 109 and Article 112".
- **s15, incomplete.** Chapters I and II still apply from 2 February 2025, but
  the two new Article 5 points and Art 5(1a), (1b) only from 2 December 2026.
- **m12, incomplete.** Same reason: "the prohibitions in Article 5" now have
  two start dates. The penalty part is unchanged.

I checked the other 42 against a word diff of every gold unit across the two
versions. The changes in Articles 2, 5, 6, 50, 99, 111 and 113 do not touch the
paragraphs those questions ask about.
