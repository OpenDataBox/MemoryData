package com.linkpoket.source.domain.aisearch.service;

import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;
import com.linkpoket.source.domain.link.repository.LinkDataAccess;
import com.linkpoket.source.domain.link.entity.Link;
import com.linkpoket.source.domain.note.repository.NoteDataAccess;
import com.linkpoket.source.domain.note.entity.Note;
import com.linkpoket.source.domain.file.repository.FileDataAccess;
import com.linkpoket.source.domain.file.entity.File;
import com.linkpoket.source.domain.postit.repository.FolderPostItDataAccess;
import com.linkpoket.source.domain.postit.entity.FolderPostIt;
import lombok.RequiredArgsConstructor;

/**
 * 특정 폴더 내에서 가장 최근에 추가되거나 편집이 일어난 최종 단건 아이템 정보(제목 및 부분 컨텐츠)를 로드하는 전담 조회 서비스입니다.
 */
@Component
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class FolderLatestItemReader {

	private final LinkDataAccess linkDataAccess;
	private final NoteDataAccess noteDataAccess;
	private final FileDataAccess fileDataAccess;
	private final FolderPostItDataAccess folderPostItDataAccess;

	/**
	 * 대상 폴더에 속해 있는 링크, 노트, 파일, 포스트잇의 최종 수정일자를 대조하여 1건을 엄선하고 "제목: / 내용: " 형태의 가벼운 식별 스니펫으로 리드합니다.
	 *
	 * @param folderId 대상 폴더 식별자
	 * @return 엄선된 1건의 식별용 한 줄 문자열 정보
	 */
	public String readLatestItemTitleAndSnippet(final Long folderId) {
		final List<Long> ids = List.of(folderId);

		final ItemSummary latestLink = linkDataAccess.findAllByFolderIdIn(ids).stream()
			.max(Comparator.comparing(Link::getEditedAt, Comparator.nullsLast(Comparator.naturalOrder())))
			.map(l -> new ItemSummary(l.getName(), l.getUrl(), l.getEditedAt()))
			.orElse(null);

		final ItemSummary latestNote = noteDataAccess.findAllByFolderIdIn(ids).stream()
			.max(Comparator.comparing(Note::getEditedAt, Comparator.nullsLast(Comparator.naturalOrder())))
			.map(n -> new ItemSummary(n.getName(), n.getContent(), n.getEditedAt()))
			.orElse(null);

		final ItemSummary latestFile = fileDataAccess.findFileEntitiesByFolderIds(ids).stream()
			.max(Comparator.comparing(File::getEditedAt, Comparator.nullsLast(Comparator.naturalOrder())))
			.map(f -> new ItemSummary(f.getName(), f.getFileUrl(), f.getEditedAt()))
			.orElse(null);

		final ItemSummary latestPostIt = folderPostItDataAccess.findAllByFolderIdIn(ids).stream()
			.max(Comparator.comparing(FolderPostIt::getEditedAt, Comparator.nullsLast(Comparator.naturalOrder())))
			.map(p -> new ItemSummary("포스트잇", p.getContent(), p.getEditedAt()))
			.orElse(null);

		final ItemSummary absoluteLatest = Stream.of(latestLink, latestNote, latestFile, latestPostIt)
			.filter(Objects::nonNull)
			.max(Comparator.comparing(ItemSummary::editedAt, Comparator.nullsLast(Comparator.naturalOrder())))
			.orElse(null);

		if (absoluteLatest == null) {
			return "";
		}

		final String titleSnippet = absoluteLatest.title() != null ? absoluteLatest.title() : "";
		final String detailSnippet = absoluteLatest.detail() != null
			? (absoluteLatest.detail().length() <= 30 ? absoluteLatest.detail() : absoluteLatest.detail().substring(0, 30))
			: "";

		return "제목: " + titleSnippet + " / 내용: " + detailSnippet;
	}
}