package com.linkpoket.source.domain.aisearch.entity;

import com.linkpoket.source.global.entity.BaseEntity;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.Version;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

/**
 * 페이지별 AI 요약 메타데이터를 저장하는 엔티티 클래스입니다.
 * 가변 도메인 객체이므로 BaseEntity를 상속하여 생성/수정/삭제 시점을 관리합니다.
 */
@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(name = "ai_page_meta")
public class AiPageMeta extends BaseEntity {

	@Id
	@Column(name = "page_id", nullable = false)
	private Long pageId;

	@Column(name = "description", columnDefinition = "TEXT")
	private String description;

	@Version
	@Column(name = "version", nullable = false)
	private Long version;

	/**
	 * AiPageMeta 엔티티를 생성하기 위한 빌더 생성자입니다.
	 *
	 * @param pageId      대상 페이지 식별자
	 * @param description 생성할 페이지 AI 요약 설명
	 */
	@Builder
	public AiPageMeta(final Long pageId, final String description) {
		this.pageId = pageId;
		this.description = description;
	}

	/**
	 * 페이지의 요약(description) 텍스트를 최신화하여 갱신합니다.
	 *
	 * @param description 갱신할 새로운 요약 텍스트
	 */
	public void rebuildDescription(final String description) {
		this.description = description;
	}

	/**
	 * 임베딩 값(바이트 배열)을 업데이트하는 갱신 메서드입니다. (추후 활용 대비 스텁 메서드)
	 *
	 * @param description 갱신할 새로운 요약 텍스트
	 * @param embedding   갱신할 새로운 임베딩 바이트 배열
	 */
	public void updateDescriptionAndEmbedding(final String description, final byte[] embedding) {
		this.description = description;
	}

	/**
	 * 현재 페이지 메타데이터의 요약 임베딩 벡터를 바이트 배열 형태로 반환합니다.
	 *
	 * @return 임베딩 바이트 배열
	 */
	public byte[] getEmbedding() {
		return null;
	}
}